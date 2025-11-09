classdef BAXOSD < ALGORITHM
% <2025> <multi/many> <real/integer/label/binary/permutation> <large>
% BAXOSD v0.3 : Bi-Axis OSD (Row–Column Dual-Drive with RVEA selection)
%
% 核心：
% - 列分组 O–S–D（结构/收敛）：目标聚类 O_k；变量列按 T 子块能量唯一指派 S_k
% - 组内线性编码器写回（C-phase）：
%       Y_k --(zscore)--> Z = Y_k*P_k  →  线性解码到 X(:,S_k)：X̂_Sk = Z*B_k（岭回归）
%       仅写回 S_k 列，半步融合 η
% - 行分组 RVEA（D-phase，多样）：参考向量扇区→锚点→稀疏度自适应→向量导向/凸组合探索
% - 环境选择：RVEA-APD（可行优先），γ 缓存，θ = FE/maxFE
%
% 示例：
%   main('-algorithm',@BAXOSD,'-problem',@DTLZ2,'-M',10,'-D',200,'-N',275,'-evaluation',1e5);

    %% ----------------- 可调参数 -----------------
    properties (Access = private)
        % O-S-D / 映射
        K           = 3;        % 目标簇数（自动钳制 <= M）
        topFrac     = 0.2;      % 唯一指派后每组保留变量列占比
        lambda      = 1e-5;     % 岭回归正则
        periodGroup = 5;        % 重分组周期
        tPeriod     = 1;        % 刷新映射/编码器周期
        emaA        = 0.2;      % T 的 EMA 平滑系数（0 关）
        % 线性编码器（组内 AE）
        kcap        = 4;        % 潜维上限
        kfrac       = 1/3;      % 潜维占比
        eta_min     = 0.3;      % 半步融合下限
        eta_max     = 0.7;      % 半步融合上限
        % C/D 比例（随 θ 增）
        rC_min      = 0.4;
        rC_max      = 0.7;
        % RVEA / D-phase
        NVscale     = 1.0;      % 参考向量数 = NVscale*N
        sigma0      = 0.15;     % 向量导向噪声基准
        beta_rho    = 1.0;      % 稀疏度放大指数
        mixAnch     = 3;        % 凸组合锚点数
        alphaDir    = 0.6;      % 向量扰动 : 锚点凸组 混合权
    end

    %% ----------------- 运行期缓存 -----------------
    properties (Access = private)
        % 全局 T（用于 S 指派）
        T      = []; muY = []; sigY = []; muX = []; sigX = [];
        % 分组
        Ok     = {};   % 目标列分组
        Sk     = {};   % 变量列分组
        owner  = [];   % 每个变量列的归属组
        % RVEA
        V      = [];   % 参考向量 (NV x M)
        gamma  = [];   % 最小夹角 (NV x 1)
        % 组内线性编码器（AE）
        Pk     = {};   % 编码矩阵（|Ok| x r_k）
        Bk     = {};   % 解码矩阵（r_k x |Sk|）
        muYk   = {}; sigYk = {};   % 组内 Y 标准化
        muXk   = {}; sigXk = {};   % 组内 X 标准化（仅 S_k 列）
    end

    methods
        function main(obj, Problem)
            %% 初始化
            Population = Problem.Initialization();

            % RVEA 参考向量初始化（一次）
            if isempty(obj.V)
                NV = max(Problem.N, round(obj.NVscale*Problem.N));
                [V0,~] = UniformPoint(NV, Problem.M);
                V0 = V0 ./ sqrt(sum(V0.^2,2)+eps);
                obj.V = V0;
                cosVV = 1 - pdist2(V0,V0,'cosine');
                cosVV(1:size(cosVV,1)+1:end) = 0;
                cosVV = max(-1,min(1,cosVV));
                obj.gamma = min(acos(cosVV),[],2);
            end

            %% 进化循环
            while obj.NotTerminated(Population)
                Y = Population.objs;
                X = local_get_dec(Population, Problem);

                % 调度参数 θ
                theta = min(1, Problem.FE / max(Problem.maxFE, eps));
                eta   = obj.eta_min + (obj.eta_max - obj.eta_min) * theta;
                rC    = obj.rC_min  + (obj.rC_max  - obj.rC_min ) * theta;
                Nc    = max(1, round(rC * Problem.N));   % C-phase 目标生成数
                Nd    = max(0, Problem.N - Nc);          % D-phase 目标生成数

                % 刷新全局线性映射 T（用于 S 指派）
                if isempty(obj.T) || mod(Problem.FE, obj.tPeriod)==0 || Problem.FE==Problem.N
                    [obj.T, obj.muY, obj.sigY, obj.muX, obj.sigX] = ...
                        fitMap_ridgeSVD_local(Y, X, obj.lambda, obj.emaA, obj.T);
                end

                % 列分组 & 唯一指派（周期）
                if isempty(obj.Ok) || mod(Problem.FE, obj.periodGroup)==0 || Problem.FE==Problem.N
                    Mk = max(2, min(obj.K, Problem.M));
                    [obj.Ok, obj.Sk, obj.owner] = group_and_assign_local(Y, obj.T, Mk, obj.topFrac);
                end

                % 刷新组内线性编码器（周期）
                if isempty(obj.Pk) || mod(Problem.FE, obj.tPeriod)==0 || Problem.FE==Problem.N
                    [obj.Pk, obj.Bk, obj.muYk, obj.sigYk, obj.muXk, obj.sigXk] = ...
                        fit_group_linearAE_all(Y, X, obj.Ok, obj.Sk, obj.kcap, obj.kfrac, obj.lambda);
                end

                % --------- C-phase：组内线性 AE 收敛生成 ---------
                OffspringC = [];
                if Nc > 0
                    candXc = [];
                    quota  = ceil(max(1, Nc / max(1,numel(obj.Ok))));
                    for k = 1:numel(obj.Ok)
                        Ik = obj.Ok{k}; Jk = obj.Sk{k};
                        if isempty(Ik) || isempty(Jk) || isempty(obj.Pk{k}) || isempty(obj.Bk{k}), continue; end
                        % 在簇内潜空间小步 DE 生成 Z'，解码回 Y_k^{new}
                        Yk = Y(:,Ik);
                        Yk_new = latent_DE_linearAE_generate(Yk, obj.Pk{k}, obj.muYk{k}, obj.sigYk{k}, quota);
                        % 仅写回 S_k 列（线性 AE 解码到 X_Sk）
                        Xhat_Sk = apply_map_sub_linearAE(Yk_new, Ik, Jk, ...
                                        obj.Pk{k}, obj.Bk{k}, obj.muYk{k}, obj.sigYk{k}, obj.muXk{k}, obj.sigXk{k}, ...
                                        X, eta, Problem.lower, Problem.upper);
                        candXc = [candXc; Xhat_Sk]; %#ok<AGROW>
                    end
                    if isempty(candXc), candXc = X; end
                    OffspringC = Problem.Evaluation(candXc);
                end

                % --------- D-phase：RVEA 行分组探索生成 ---------
                OffspringD = [];
                if Nd > 0
                    [Y_dirs] = rvea_generate_dirs(Y, obj.V, obj.mixAnch, obj.alphaDir, obj.sigma0, obj.beta_rho, theta, Nd);
                    % 对每个列簇分别写回（避免串扰），叠加候选
                    candXd = [];
                    for k = 1:numel(obj.Ok)
                        Ik = obj.Ok{k}; Jk = obj.Sk{k};
                        if isempty(Ik) || isempty(Jk) || isempty(obj.Pk{k}) || isempty(obj.Bk{k}), continue; end
                        Yk_new = Y_dirs(:, Ik);
                        Xhat_Sk = apply_map_sub_linearAE(Yk_new, Ik, Jk, ...
                                        obj.Pk{k}, obj.Bk{k}, obj.muYk{k}, obj.sigYk{k}, obj.muXk{k}, obj.sigXk{k}, ...
                                        X, eta, Problem.lower, Problem.upper);
                        candXd = [candXd; Xhat_Sk]; %#ok<AGROW>
                    end
                    if isempty(candXd), candXd = X; end
                    OffspringD = Problem.Evaluation(candXd);
                end

                % --------- 环境选择：RVEA-APD（可行优先） ---------
                Population = env_select_rvea_apd([Population, OffspringC, OffspringD], ...
                                                  Problem.N, obj.V, obj.gamma, theta);
            end
        end
    end
end

%% ========================= 局部函数 =========================

% ---- 安全获取决策矩阵 ----
function Dec = local_get_dec(Population, Problem)
try
    Dec = PopDec(Population); if isempty(Dec), error('empty'); end
catch
    if isprop(Population,'decs')
        Dec = Population.decs;
    elseif isprop(Population,'dec')
        Dec = vertcat(Population.dec);
    else
        N = numel(Population); D = Problem.D;
        lb = repmat(Problem.lower, N,1); ub = repmat(Problem.upper, N,1);
        Dec = lb + rand(N,D).*(ub-lb);
    end
end
end

% ---- 全局 T：岭回归 + SVD（zscore 域），可 EMA ----
function [T, muY, sigY, muX, sigX] = fitMap_ridgeSVD_local(Yraw, Xraw, lambda, emaA, Told)
muY = mean(Yraw,1);  sigY = std(Yraw,[],1);  sigY(sigY==0)=1;
muX = mean(Xraw,1);  sigX = std(Xraw,[],1);  sigX(sigX==0)=1;
Y = (Yraw - muY)./sigY;  X = (Xraw - muX)./sigX;
[U,S,V] = svd(Y,'econ'); s = diag(S);
invTerm = diag(s./(s.^2 + lambda));
Tnew = V*invTerm*(U')*X;                 % (M x D)
if nargin>=5 && ~isempty(Told) && emaA>0
    T = (1-emaA)*Told + emaA*Tnew;
else
    T = Tnew;
end
end

% ---- 目标聚类 + 变量唯一指派（按 T 子块能量） ----
function [Ok, Sk, owner] = group_and_assign_local(Y, T, K, topFrac)
[~,M] = size(Y);
K = max(2, min(K, M));
Yc = Y - mean(Y,1); Yc = Yc ./ max(std(Yc,[],1), 1e-12);
try
    lab = kmeans(Yc', K, 'Distance','cosine', 'Replicates',5, 'MaxIter',100);
catch
    C = Yc(:, randperm(M, K));
    for it=1:8
        cosYX = 1 - pdist2(Yc', C','cosine'); cosYX(isnan(cosYX))=-1;
        [~, lab] = max(cosYX, [], 2);
        for k=1:K
            idx = find(lab==k);
            if ~isempty(idx), C(:,k) = mean(Yc(:,idx),2); end
        end
    end
end
Ok = cell(1,K);
for k=1:K, Ok{k} = find(lab==k)'; end

D = size(T,2);
energy = zeros(K, D);
for k=1:K
    Ik = Ok{k}; if isempty(Ik), continue; end
    energy(k,:) = sqrt(sum(T(Ik,:).^2,1));      % 子块列范数
end
[~,owner] = max(energy, [], 1);

Sk = cell(1,K);
p = max(1, round(topFrac * D));
for k=1:K
    cols = find(owner==k);
    if isempty(cols), Sk{k} = []; continue; end
    [~,ord] = sort(energy(k,cols),'descend');
    Sk{k} = cols(1:min(p, numel(cols)));
end
end

% ---- 拟合所有簇的线性编码器（P_k,B_k） ----
function [Pk, Bk, muYk, sigYk, muXk, sigXk] = fit_group_linearAE_all(Y, X, Ok, Sk, kcap, kfrac, lambda)
K = numel(Ok);
Pk = cell(1,K); Bk = cell(1,K);
muYk = cell(1,K); sigYk = cell(1,K);
muXk = cell(1,K); sigXk = cell(1,K);
for k=1:K
    Ik = Ok{k}; Jk = Sk{k};
    if isempty(Ik) || isempty(Jk)
        Pk{k}=[]; Bk{k}=[]; muYk{k}=[]; sigYk{k}=[]; muXk{k}=[]; sigXk{k}=[];
        continue;
    end
    Yk = Y(:,Ik); Xk = X(:,Jk);
    [P,B,muYc,sigYc,muXc,sigXc] = fit_group_linearAE(Yk, Xk, kcap, kfrac, lambda);
    Pk{k}=P; Bk{k}=B; muYk{k}=muYc; sigYk{k}=sigYc; muXk{k}=muXc; sigXk{k}=sigXc;
end
end

% ---- 单簇线性 AE：P（编码，PCA） + B（岭：Z->X_Sk） ----
function [P, B, muYc, sigYc, muXc, sigXc] = fit_group_linearAE(Yk, Xk, kcap, kfrac, lambda)
muYc = mean(Yk,1); sigYc = std(Yk,[],1); sigYc(sigYc==0)=1;
muXc = mean(Xk,1); sigXc = std(Xk,[],1); sigXc(sigXc==0)=1;
Yz = (Yk - muYc)./sigYc;   % N x mk
Xz = (Xk - muXc)./sigXc;   % N x |Sk|
[~,~,V] = svd(Yz,'econ');  % mk x mk
mk = size(Yk,2);
r  = max(1, min([size(V,2), kcap, floor(mk*kfrac)]));
P  = V(:,1:r);             % mk x r
Z  = Yz * P;               % N x r
B  = (Z.'*Z + lambda*eye(r)) \ (Z.'*Xz);   % r x |Sk|
end

% ---- C-phase：簇内潜空间 DE 生成 Y_k^{new} ----
function Yk_new = latent_DE_linearAE_generate(Yk, P, muYc, sigYc, Nout)
[N, mk] = size(Yk);
if mk==0 || isempty(P)
    Yk_new = repmat(mean(Yk,1), max(Nout,1), 1); return;
end
Yz = (Yk - muYc)./sigYc;              % N x mk
Z  = Yz * P;                          % N x r
r  = size(P,2);
F=0.7; Cr=0.9;
Znew = zeros(max(Nout,1), r);
if N >= 3
    for i=1:size(Znew,1)
        rr = randperm(N,3);
        v  = Z(rr(1),:) + F*(Z(rr(2),:) - Z(rr(3),:));
        u  = Z(randi(N),:);
        jrand = randi(r);
        mask = (rand(1,r)<=Cr); mask(jrand)=true;
        u(mask)=v(mask);
        Znew(i,:) = u;
    end
else
    Znew = repmat(mean(Z,1), max(Nout,1), 1) + 0.1*randn(max(Nout,1), r);
end
Yk_new = (Znew * P.') .* sigYc + muYc;   % Nout x mk
end

% ---- 写回：线性 AE（仅 S_k 列，半步融合） ----
function Xhat = apply_map_sub_linearAE(Yk_new, Ik, Jk, P, B, muYc, sigYc, muXc, sigXc, Xbase, eta, lower, upper)
Nnew = size(Yk_new,1);
D    = size(Xbase,2);
% 编码到 Z
Yz_new = (Yk_new - muYc)./sigYc;       % Nnew x mk
Znew   = Yz_new * P;                    % Nnew x r
% 解码到 X_Sk（zscore 域）
Xz_pred = Znew * B;                     % Nnew x |Jk|
Xpred_Sk = Xz_pred .* sigXc + muXc;     % 反标准化
% 准备基底
Nbase = size(Xbase,1);
rep  = ceil(Nnew / Nbase);
Xhat = repmat(Xbase, rep, 1); Xhat = Xhat(1:Nnew, :);
% 半步融合，仅 S_k
Xhat(:,Jk) = (1-eta).*Xhat(:,Jk) + eta.*Xpred_Sk;
% 边界
lo = lower(:)'; up = upper(:)';
Xhat = min(max(Xhat, lo.*ones(Nnew,1)), up.*ones(Nnew,1));
end

% ---- 生成 D-phase 的全维 Y_dir（RVEA 行分组引导） ----
function [Y_dirs] = rvea_generate_dirs(Y, V, mixAnch, alphaDir, sigma0, beta_rho, theta, Nd)
N = size(Y,1); M = size(Y,2); NV = size(V,1);
mins = min(Y,[],1); Ysh = Y - mins;
cos_xv = 1 - pdist2(Ysh, V, 'cosine');
cos_xv = max(-1,min(1,cos_xv)); cos_xv(isnan(cos_xv))=-1;
Angle  = acos(cos_xv);
[~, assoc] = min(Angle, [], 2);

rowGroup = cell(1,NV); counts = zeros(1,NV);
for i=1:NV, rowGroup{i} = find(assoc==i)'; counts(i)=numel(rowGroup{i}); end
rho = counts ./ max(mean(counts)+eps, eps);      % 稀疏度

% 扇区锚点（角度最小）
anchors = zeros(NV,1);
for i=1:NV
    if counts(i)>0
        [~,loc] = min(Angle(rowGroup{i}, i));
        anchors(i) = rowGroup{i}(loc);
    else
        anchors(i) = 0;
    end
end

% 将 Nd 平均分配到扇区（简化；可按 rho 自适应）
qi_all = max(1, floor(Nd / max(NV,1)));
Y_dirs = zeros(0,M);

% 预取全部锚点矩阵（用于凸组合）
haveAnch = anchors>0;
Yanch = Y(anchors(haveAnch), :);

for i=1:NV
    qi = qi_all; if qi<=0, continue; end
    sig_i = sigma0 * max(0.05, 1-theta) * (max(mean(rho),eps)/(rho(i)+eps))^beta_rho;
    % 参考向量导向噪声尺度
    scaleR = median(sqrt(sum(Ysh.^2,2))+eps);

    Y_i = zeros(qi, M);
    for t=1:qi
        if anchors(i)>0, y0 = Y(anchors(i),:); else, y0 = mean(Y,1); end
        % 向量导向扰动
        step = randn(1) * sig_i * scaleR;
        y_vec = y0 + step * V(i,:);
        % 锚点凸组合
        if size(Yanch,1) >= mixAnch
            cos_a = 1 - pdist2(Yanch - mins, V(i,:), 'cosine');
            cos_a = max(-1,min(1,cos_a)); cos_a(isnan(cos_a))=-1;
            [~,ord] = sort(acos(cos_a), 'ascend');
            pick = Yanch(ord(1:mixAnch), :);
            w = sparse_dirichlet(1, mixAnch, 0.25);   % 稀疏权
            y_mix = w*pick;
        else
            y_mix = y0;
        end
        Y_i(t,:) = alphaDir*y_vec + (1-alphaDir)*y_mix;
    end
    Y_dirs = [Y_dirs; Y_i]; %#ok<AGROW>
end

if isempty(Y_dirs)
    Y_dirs = repmat(mean(Y,1), max(Nd,1), 1);
end
% 截断为 Nd 条
if size(Y_dirs,1) > Nd
    Y_dirs = Y_dirs(1:Nd,:);
end
end

% ---- 环境选择：RVEA-APD（可行优先） ----
function Population = env_select_rvea_apd(PopAll, N, V, gamma, theta)
if isempty(PopAll), Population = PopAll; return; end
FrontNo = NDSort(PopAll.objs, 1);
Pop = PopAll(FrontNo==1);
if isempty(Pop)
    fmin = min(FrontNo); Pop = PopAll(FrontNo==fmin);
end
if isempty(Pop)
    Population = PopAll(1:min(N, length(PopAll))); return;
end

Nf = length(Pop);
PopObj = Pop.objs;
[~, M] = size(PopObj);

mins = min(PopObj,[],1);
PopObj = PopObj - mins;

CV = sum(max(0, Pop.cons), 2);
if isempty(CV) || numel(CV)~=Nf, CV = zeros(Nf,1); end

cos_xv = 1 - pdist2(PopObj, V, 'cosine');
cos_xv = max(-1,min(1,cos_xv)); cos_xv(isnan(cos_xv))=-1;
Angle  = acos(cos_xv);
[~, associate] = min(Angle, [], 2);

NV = size(V,1); Next = zeros(1,NV);
for i = unique(associate(:))'
    idx = find(associate==i);
    idxFea = idx(CV(idx)==0);
    idxInf = idx(CV(idx)~=0);
    if ~isempty(idxFea)
        gi = max(gamma(i), 1e-12);
        ang = Angle(idxFea, i);
        normf = sqrt(sum(PopObj(idxFea,:).^2,2));
        APD = (1 + M*theta*(ang/gi)) .* normf;
        [~,best] = min(APD); Next(i) = idxFea(best);
    elseif ~isempty(idxInf)
        [~,best] = min(CV(idxInf)); Next(i) = idxInf(best);
    end
end

pick = Next(Next~=0);
if numel(pick) ~= N
    selected = false(1,Nf); selected(pick)=true;
    if numel(pick) < N
        rem = find(~selected); feas = rem(CV(rem)==0);
        if ~isempty(feas)
            nf = sqrt(sum(PopObj(feas,:).^2,2));
            [~,ord] = sort(nf,'ascend');
            need = min(N-numel(pick), numel(feas));
            pick = [pick, feas(ord(1:need))]; %#ok<AGROW>
            selected(pick)=true;
        end
    end
    if numel(pick) < N
        rem = find(~selected); infs = rem(CV(rem)~=0);
        if ~isempty(infs)
            [~,ord] = sort(CV(infs),'ascend');
            need = min(N-numel(pick), numel(infs));
            pick = [pick, infs(ord(1:need))]; %#ok<AGROW>
        end
    end
end

pick = pick(:)'; pick = pick(isfinite(pick));
pick = pick(pick==floor(pick)); pick = pick(pick>0 & pick<=Nf);
pick = unique(pick,'stable');
if isempty(pick)
    pick = 1:min(N,Nf);
elseif numel(pick) < N
    rep = pick; if isempty(rep), rep = 1:min(N,Nf); end
    while numel(pick) < N
        pick(end+1) = rep(mod(numel(pick), numel(rep))+1); %#ok<AGROW>
    end
elseif numel(pick) > N
    pick = pick(1:N);
end
Population = Pop(pick);
end

% ---- 稀疏狄利克雷权（简易） ----
function W = sparse_dirichlet(N, K, conc)
r = rand(N,K) .^ (1/max(conc,1e-3));
W = r ./ sum(r,2);
end
