function model = trainDAE_walkback(Y, varargin)
% TRAINDAE_WALKBACK  在目标空间 Y 上训练去噪自编码器（广义 DAE）+ Walkback
% 用法（两种都支持）：
%   model = trainDAE_walkback(Y, optsStruct);             % 传一个 struct
%   model = trainDAE_walkback(Y, 'latentDim',8, ...);     % Name-Value 形式
%
% 依赖：Deep Learning Toolbox (dlnetwork, dlarray, forward, adamupdate)

% ---------- 0) 解析参数（兼容 struct 与 Name-Value） ----------
defaults = struct( ...
    'latentDim',          0, ...        % 0 表示自动设为 ~1.2×M
    'noiseScales',        [], ...
    'noiseSigmaRatio',    [0.05 0.15], ...
    'maskProb',           0.0, ...
    'walk_p',             0.5, ...
    'walk_trajs',         1, ...
    'epochs',             50, ...
    'batchSize',          128, ...
    'learnRate',          1e-3, ...
    'weightDecay',        0.0, ...
    'verbose',            true, ...
    'gpu',                false ...
);
opts = defaults;

if nargin >= 2
    if numel(varargin)==1 && isstruct(varargin{1})
        % 传的是 struct
        user = varargin{1};
        fns = fieldnames(user);
        for i = 1:numel(fns)
            opts.(fns{i}) = user.(fns{i});
        end
    else
        % Name-Value 对
        if mod(numel(varargin),2)~=0
            error('Name-Value 参数必须成对提供。');
        end
        for i = 1:2:numel(varargin)
            name = varargin{i};
            val  = varargin{i+1};
            if ~ischar(name) && ~isstring(name)
                error('Name-Value 中的 Name 必须是字符/字符串。');
            end
            name = char(name);
            opts.(name) = val;
        end
    end
end

% ---------- 1) 初始化 ----------
[N, M] = size(Y);
Y = single(Y);
if opts.latentDim <= 0
    opts.latentDim = max(6, round(1.2*M));
end

% ---------- 2) 网络结构（合体：编码+解码） ----------
layers = [
    featureInputLayer(M, "Name","in")
    fullyConnectedLayer(128, "Name","enc_fc1")
    reluLayer("Name","enc_relu1")
    fullyConnectedLayer(64, "Name","enc_fc2")
    reluLayer("Name","enc_relu2")
    fullyConnectedLayer(opts.latentDim, "Name","bottleneck")
    reluLayer("Name","enc_relu3")
    fullyConnectedLayer(64, "Name","dec_fc1")
    reluLayer("Name","dec_relu1")
    fullyConnectedLayer(128, "Name","dec_fc2")
    reluLayer("Name","dec_relu2")
    fullyConnectedLayer(M, "Name","out")   % 线性输出 ~ 高斯均值
];
lgraph = layerGraph(layers);
dlnet  = dlnetwork(lgraph);

% ---------- 3) 多尺度噪声池 ----------
stdY = std(Y,0,1);   % 1 x M
if isempty(opts.noiseScales)
    r = opts.noiseSigmaRatio;  % [low high]
    sigmaSmall = max(1e-6, r(1) * stdY);
    sigmaMid   = max(1e-6, r(2) * stdY);
    noisePool  = cat(1, sigmaSmall, sigmaMid);   % 2 x M
else
    noisePool  = single(opts.noiseScales);       % K x M
end

% ---------- 4) 训练循环（Adam） ----------
trailingAvg = []; trailingAvgSq = []; iter = 0;
numItersPerEpoch = ceil(N/opts.batchSize);
if opts.verbose
    fprintf("[DAE] N=%d, M=%d, latent=%d, epochs=%d, batch=%d, walk_p=%.2f\n", ...
        N,M,opts.latentDim,opts.epochs,opts.batchSize,opts.walk_p);
end
useGPU = opts.gpu && canUseGPU;

for epoch = 1:opts.epochs
    order = randperm(N);
    for it = 1:numItersPerEpoch
        iter = iter + 1;
        idx  = order((it-1)*opts.batchSize+1 : min(it*opts.batchSize, N));
        Yb   = Y(idx, :);   % B x M

        % ---- 基础腐蚀 ----
        sigmas   = pickSigmas(noisePool, size(Yb,1));   % B x M
        Ytil     = corruptY(Yb, sigmas, opts.maskProb); % B x M
        Y_all    = Yb;      % (B+aug) x M
        Ytil_all = Ytil;

        % ---- Walkback 扩充 ----
        for t = 1:opts.walk_trajs
            Ystar = Yb;
            cont  = rand(size(Yb,1),1) < opts.walk_p;
            while any(cont)
                sigWB = pickSigmas(noisePool, sum(cont));
                YtWB  = corruptY(Ystar(cont,:), sigWB, opts.maskProb);
                % 去噪一步（用当前网络）
                Yrec  = forward_decode(dlnet, YtWB, useGPU);
                % 记录沿途坏点
                Y_all    = [Y_all;    Yb(cont,:)]; %#ok<AGROW>
                Ytil_all = [Ytil_all; YtWB];       %#ok<AGROW>
                % 更新 walk 状态
                Ystar(cont,:) = Yrec;
                cont = rand(size(Yb,1),1) < opts.walk_p;
            end
        end

        % ---- 前向 + 反向 + 更新 ----
        dlX = dlarray(single(Ytil_all)', 'CB');  % M x B'
        dlT = dlarray(single(Y_all)'   , 'CB');  % M x B'
        if useGPU, dlX = gpuArray(dlX); dlT = gpuArray(dlT); end

        [gradients, loss] = dlfeval(@modelGradients, dlnet, dlX, dlT, opts.weightDecay);
        [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, gradients, ...
            trailingAvg, trailingAvgSq, iter, opts.learnRate);

        if opts.verbose && mod(it, max(1,floor(numItersPerEpoch/5)))==0
            fprintf("  epoch %d/%d  iter %d/%d  pairs=%d  loss=%.6f\n", ...
                epoch,opts.epochs,it,numItersPerEpoch, size(Y_all,1), gather(extractdata(loss)));
        end
    end
end

% ---------- 5) 导出模型 ----------
model.net        = dlnet;
model.noisePool  = noisePool;
model.maskProb   = opts.maskProb;
model.walk_p     = opts.walk_p;
model.decodeFcn  = @(Ytil) forward_decode(dlnet, Ytil, useGPU);
model.corruptFcn = @(Yin)  corruptY(Yin, pickSigmas(noisePool, size(Yin,1)), opts.maskProb);

end % ===== 主函数结束 =====


% ====== 工具函数们 ======
function sig = pickSigmas(noisePool, B)
K = size(noisePool,1);
idx = randi(K,[B,1]);
sig = noisePool(idx,:); % B x M
end

function Ytil = corruptY(Y, sigmas, maskProb)
noise = randn(size(Y),'like',Y) .* sigmas;
Ytil  = Y + noise;
if maskProb>0
    mask = rand(size(Y),'like',Y) < maskProb;
    Ytil(mask) = Y(mask);   % “轻掩蔽”：保留原值
end
end

function Yrec = forward_decode(dlnet, Ytil, useGPU)
% 用当前网络做一次去噪预测（避免对带标签 dlarray 直接转置）
dlX = dlarray(single(Ytil)', 'CB');      % M x B
if useGPU && canUseGPU, dlX = gpuArray(dlX); end
Yhat = forward(dlnet, dlX);              % M x B (带标签)
Yrec = gather(extractdata(Yhat))';       % 先解包成数值，再转置成 B x M
end

function [gradients, loss] = modelGradients(dlnet, dlX, dlT, weightDecay)
Yhat = forward(dlnet, dlX);
err  = Yhat - dlT;                 % M x B
mse  = mean(err.^2,'all');
if weightDecay>0
    L = dlnet.Learnables; s = dlarray(0,'SS');
    for i = 1:height(L), v = L.Value{i}; s = s + sum(v.^2,'all'); end
    loss = mse + weightDecay*s;
else
    loss = mse;
end
gradients = dlgradient(loss, dlnet.Learnables);
end
