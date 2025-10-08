function Ynew = sampleY_walkback(model, Yseed, varargin)
% SAMPLEY_WALKBACK  用“腐蚀 ↔ 去噪”链从已训练的 DAE 采样新目标向量
% 用法（两种都支持）：
%   Ynew = sampleY_walkback(model, Yseed, optsStruct)
%   Ynew = sampleY_walkback(model, Yseed, 'stepsMin',5,'stepsMax',20,'condSigma',0.0)

% -------- 参数解析（兼容 struct 与 Name-Value） --------
opts = struct('stepsMin',5, 'stepsMax',20, 'condSigma',0.0);
if ~isempty(varargin)
    if numel(varargin)==1 && isstruct(varargin{1})
        user = varargin{1};
        fns = fieldnames(user);
        for i = 1:numel(fns), opts.(fns{i}) = user.(fns{i}); end
    else
        if mod(numel(varargin),2)~=0
            error('Name-Value 参数必须成对提供。');
        end
        for i = 1:2:numel(varargin)
            name = varargin{i}; val = varargin{i+1};
            if ~ischar(name) && ~isstring(name)
                error('Name 必须是字符/字符串。');
            end
            opts.(char(name)) = val;
        end
    end
end

% -------- 校验 & 取模型字段 --------
assert(isfield(model,'net') && isfield(model,'decodeFcn') && isfield(model,'noisePool'), ...
    'model 不是由 trainDAE_walkback 返回的结构体。');
if ~isfield(model,'maskProb'), model.maskProb = 0; end

J = size(Yseed,1);
M = size(Yseed,2);
Y = single(Yseed);

% -------- 采样链：每条链走 [stepsMin, stepsMax] 步 --------
for j = 1:J
    S = randi([max(0,opts.stepsMin), max(opts.stepsMin, opts.stepsMax)]);
    y = Y(j,:);
    for s = 1:S
        % 1) 腐蚀：随机挑一档噪声尺度
        sigmas = model.noisePool(randi(size(model.noisePool,1)),:);
        ytil = corruptY(y, sigmas, model.maskProb);
        % 2) 去噪
        y = model.decodeFcn(ytil);
        % 3) 可选：在去噪均值附近加一点小噪（多样性）
        if opts.condSigma > 0
            y = y + randn(1,M,'like',y) .* (opts.condSigma * sigmas);
        end
    end
    Y(j,:) = y;
end

Ynew = double(Y);
end

% ---- 局部腐蚀函数 ----
function Ytil = corruptY(Y, sigmas, maskProb)
noise = randn(size(Y),'like',Y) .* sigmas;
Ytil = Y + noise;
if maskProb>0
    mask = rand(size(Y),'like',Y) < maskProb;
    Ytil(mask) = Y(mask);
end
end
