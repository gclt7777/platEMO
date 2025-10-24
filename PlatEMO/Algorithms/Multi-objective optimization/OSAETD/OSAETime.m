classdef OSAETime < ALGORITHM
% <2025> <multi/many> <real/integer/label/binary/permutation>
% OS_AE_T : Objective-space Grouping + AE(PCA) + Linear T write-back
% - 仅用线性 T 做“变量指派 + 写回”，写回时只改各自 S_k 列（避免串扰）
% - T 拟合采用岭回归；支持子采样与低秩近似；可控更新频率；可选 EMA 平滑
% - 组内 AE (PCA) 支持缓存与按周期复用

    methods
        function main(obj, Problem)
            % 参数（可命令行覆盖）
            %   K, topFrac        : 目标簇数；每组取前 topFrac 变量形成 S_k
            %   lambda            : 拟合 T 的岭值
            %   eta               : 决策半步写回系数
            %   kcap, kfrac       : 组内 PCA 潜维上限与比例
            %   F,Cr              : 潜空间 DE 参数
            %   period            : 每隔多少代重分组/重指派
            %   emaA              : T 的 EMA 平滑系数（0 关闭）
            %   tPeriod           : T 的更新周期（代数）；=1 表示每代
            %   mapSampleFrac     : 拟合 T 的子采样比例 (0,1]，例如 0.3
            %   mapRankCap        : 拟合 T 的低秩近似上限（0 表示不裁剪）
            %   pcaPeriod         : 组内 PCA/标准化缓存的复用周期（代数）
            [K,topFrac,lambda,eta,kcap,kfrac,F,Cr,period,emaA, ...
             tPeriod,mapSampleFrac,mapRankCap,pcaPeriod] = ...
                obj.ParameterSet(3,0.2,1e-5,0.5,4,1/3,0.7,0.9,5,0.0, ...
                                 3,         0.3,          6,        5);

            % 初始化
            Population = Problem.Initialization();
            gen = 1;  T_prev = [];

            % —— 首次：分组 + T + 指派（使用子采样/低秩近似选项）
            mapOpts.subFrac = mapSampleFrac;
            mapOpts.rankCap = mapRankCap;
            [O_groups, S_groups, T] = group_and_assign_T(Problem, Population, K, topFrac, lambda, mapOpts);
            if emaA>0, T_prev = T; end

            % —— 组内 PCA 缓存（按分组个数动态维护）
            pcaCache = cell(1, numel(O_groups));    % 每组存 W,mu,sg,gen,dim

            while obj.NotTerminated(Population)
                X = Population.decs;   % N×D
                Y = Population.objs;   % N×M
                OffDec = X;

                % —— 重分组/重指派
                if gen==1 || mod(gen-1,period)==0
                    [O_groups, S_groups, T_new] = group_and_assign_T(Problem, Population, K, topFrac, lambda, mapOpts);
                    % 分组变化时重置/对齐 PCA 缓存
                    if numel(pcaCache) ~= numel(O_groups), pcaCache = cell(1,numel(O_groups)); end
                else
                    T_new = T; % 默认继承
                    % —— T 更新（与重分组解耦，由 tPeriod 控制）
                    if mod(gen-1,tPeriod)==0
                        T_new = fitMap_ridgeSVD_fast(Y, X, lambda, mapOpts);
                    end
                end

                % —— 可选 EMA 平滑 T
                if emaA>0 && ~isempty(T_prev)
                    T = (1-emaA)*T_prev + emaA*T_new;
                    T_prev = T;
                else
                    T = T_new;  T_prev = T_new;
                end

                % —— 逐目标组：AE(PCA) 生成 + 用子块 T(Ok,Sk) 半步写回
                for k = 1:numel(O_groups)
                    Ok = O_groups{k};      % 该组目标索引
                    Sk = S_groups{k};      % 该组写回的决策索引
                    if isempty(Ok) || isempty(Sk), continue; end

                    % 组内 AE: PCA + 一轮 DE（支持缓存与按 pcaPeriod 复用）
                    [Yk_new, pcaCache{k}] = ae_pca_generate_cached(Y(:,Ok), kcap, kfrac, F, Cr, pcaCache{k}, gen, pcaPeriod);

                    % 线性写回（只用子块）
                    Xhat_Sk      = Yk_new * T(Ok,Sk);                 % N×|Sk|
                    OffDec(:,Sk) = (1-eta)*OffDec(:,Sk) + eta*Xhat_Sk;
                end

                % 裁剪 + 评估 + 环境选择（保持原风格：NDSort+角度截断）
                OffDec    = min(max(OffDec, Problem.lower), Problem.upper);
                Offspring = Problem.Evaluation(OffDec);
                Population = env_select_local([Population, Offspring], Problem.N);

                gen = gen + 1;
            end
        end
    end
end

%% ========================= 内联工具 =========================
function [O_groups, S_groups, T] = group_and_assign_T(Problem, Population, K, topFrac, lambda, mapOpts)
X = Population.decs; Y = Population.objs;
[~,D] = size(X);  M = size(Y,2);

% K 钳制
if isempty(K) || ~isscalar(K) || K<1, K = min(3,M); end
if M==1, K_eff=1; else, K_eff = min(K, max(1, M-1)); end

% 分组：优先外部函数，否则内置
try
    [Og,~,~] = OS_GroupByObjective(Population,'K',K_eff);
    O_groups = Og;
catch
    O_groups = os_group_by_objective_builtin(Y, K_eff);
end

% 拟合 T（子采样/低秩近似）
T = fitMap_ridgeSVD_fast(Y, X, lambda, mapOpts);

% 唯一指派并组内取 topFrac
Keff = numel(O_groups);
energy = zeros(Keff,D);
for k = 1:Keff
    Ok = O_groups{k}; if isempty(Ok), continue; end
    Tk = T(Ok,:);                         % |Ok|×D
    energy(k,:) = sqrt(sum(Tk.^2,1));
end
[~, owner] = max(energy, [], 1);

S_groups = cell(1,Keff);
for k = 1:Keff
    idx = find(owner==k);
    if isempty(idx), S_groups{k} = []; continue; end
    e = energy(k,idx);
    [~,ord] = sort(e,'descend');
    topK = max(1, round(topFrac*numel(idx)));
    S_groups{k} = idx(ord(1:topK));
end
end

%% —— 更快的岭回归 T 拟合（子采样 + 低秩近似） ——
function T = fitMap_ridgeSVD_fast(Y, X, lambda, opts)
% opts.subFrac ∈ (0,1]；opts.rankCap >=0（0 表示不裁剪）
if nargin<4 || isempty(opts), opts = struct(); end
if ~isfield(opts,'subFrac') || isempty(opts.subFrac), opts.subFrac = 1.0; end
if ~isfield(opts,'rankCap') || isempty(opts.rankCap), opts.rankCap = 0; end
[N,~] = size(Y);
if opts.subFrac < 1
    nSub = max(ceil(opts.subFrac*N), max(30, min(N, 5*size(Y,2))));
    idx  = randperm(N, nSub);
    Ys = Y(idx,:);  Xs = X(idx,:);
else
    Ys = Y;         Xs = X;
end
% 标准化（按子样本）
muY = mean(Ys,1);  sigY = std(Ys,0,1); sigY(sigY==0)=1;
muX = mean(Xs,1);  sigX = std(Xs,0,1); sigX(sigX==0)=1;
Yz  = (Ys - muY)./sigY;
Xz  = (Xs - muX)./sigX;

% 低秩近似：对 C = Yz'Yz 做特征分解（M 通常较小）
C = (Yz.'*Yz);                               % M×M
[C_V, C_D] = eig((C+C.')/2);
sig2 = max(real(diag(C_D)), 0);
[ sig2, ord ] = sort(sig2,'descend');
V = C_V(:,ord);
if opts.rankCap>0
    r = min(opts.rankCap, size(V,2));
    V = V(:,1:r); sig2 = sig2(1:r);
end
% T = V * diag(1./(sig^2+lambda)) * V' * (Yz' * Xz)
S = Yz.' * Xz;                                % M×D
invTerm = V * diag(1./(sig2 + lambda)) * V.'; % M×M
Tz = invTerm * S;                              % M×D （标准化域下）

% 输出：保持与原写回流程一致（Yk_new * T(Ok,Sk) 已在“标准化外”域）
T = Tz;
end

%% —— 组内 AE (PCA) 缓存与生成 ——
function [Yk_new, cache] = ae_pca_generate_cached(Yk, kcap, kfrac, F, Cr, cache, gen, pcaPeriod)
if nargin<6 || isempty(cache), cache = struct(); end
recalc = true;
d = size(Yk,2);
% 需要重算的条件：无缓存；维度变了；到达周期
if isfield(cache,'dim') && cache.dim==d && isfield(cache,'gen') && (mod(gen-1,pcaPeriod)~=0)
    % 复用缓存参数
    if all(isfield(cache, {'W','mu','sg'})) && ~isempty(cache.W)
        recalc = false;
    end
end

if recalc
    [Ykz, mu, sg] = safe_zscore(Yk);
    r  = max(1, min([d, max(1, ceil(d*kfrac)), kcap]));
    W  = pca_basis(Ykz, r);
    cache.W = W; cache.mu = mu; cache.sg = sg; cache.gen = gen; cache.dim = d;
else
    W = cache.W; mu = cache.mu; sg = cache.sg;
    Ykz = (Yk - mu)./sg;
end

Z  = Ykz * W;
Z2 = latent_DE(Z, F, Cr);
Yk_new = (Z2 * W') .* sg + mu;
end

%% —— 聚类与选择等工具（保持原风格） ——
function O_groups = os_group_by_objective_builtin(Y, K)
[~,M] = size(Y);
if M==1 || K<=1, O_groups = {1:M}; return; end
K = min(K, max(1, M-1));
Yz = zscore(Y,0,1); Yz(:,any(isnan(Yz),1))=0;
nrm = sqrt(sum(Yz.^2,1)); nrm(nrm==0)=1;
U = Yz ./ nrm;
S = U.'*U;  S(1:M+1:end) = 0;
A = max(S,0);  A = A - diag(diag(A));
d = sum(A,2);  Dm = diag(d + eps);
Lsym = eye(M) - Dm^(-1/2)*A*Dm^(-1/2); Lsym = (Lsym + Lsym.')/2;
[V,E] = eig(Lsym); [~,ord] = sort(diag(E),'ascend');
H  = V(:,ord(1:K));
Hn = H ./ max(sqrt(sum(H.^2,2)), eps); Hn(~isfinite(Hn)) = 0;
opts = statset('MaxIter',200,'Display','off');
repl = max(5, min(10, M-1));
lbl = kmeans(Hn, K, 'Replicates', repl, 'Options', opts);
O_groups = cell(1,K); for k = 1:K, O_groups{k} = find(lbl==k).'; end
end

function [Z,mu,sg] = safe_zscore(X)
mu = mean(X,1);
sg = std(X,0,1); sg(sg==0) = 1;
Z = (X - mu) ./ sg;
end

function W = pca_basis(X, k)
[~,~,V] = svd(X,'econ'); W = V(:,1:k);
end

function Znew = latent_DE(Z, F, Cr)
[N,k] = size(Z);
Znew = Z;
if N<4 || k==0, return; end
idx = 1:N;
for i = 1:N
    r = idx; r(i) = [];
    if numel(r)<3, Znew(i,:) = Z(i,:); continue; end
    r = r(randperm(numel(r),3));
    v = Z(r(1),:) + F*(Z(r(2),:) - Z(r(3),:));
    jrand = randi(k);
    mask = (rand(1,k) < Cr); mask(jrand) = true;
    u = Z(i,:); u(mask) = v(mask);
    Znew(i,:) = u;
end
end

function Population = env_select_local(Population, N)
PopObj = Population.objs;
[FrontNo, MaxFNo] = NDSort(PopObj, N);
Next = FrontNo < MaxFNo;
Last = find(FrontNo == MaxFNo);
K = N - sum(Next);
if K > 0
    Choose = truncation_angle(PopObj(Last,:), K);
    Next(Last(Choose)) = true;
end
Population = Population(Next);
end

function Choose = truncation_angle(PopObj, K)
fmax = max(PopObj,[],1); fmin = min(PopObj,[],1);
span = fmax - fmin; span(span==0) = 1;
P = (PopObj - fmin) ./ span;
nrm = sqrt(sum(P.^2,2)); nrm(nrm==0)=1; U = P ./ nrm;
Cosine = U*U.'; Cosine(1:size(Cosine,1)+1:end)=0;
Choose = false(1,size(P,1));
[~,extreme] = max(P,[],1); Choose(extreme)=true;
if sum(Choose) > K
    sel = find(Choose); Choose=false(1,size(P,1));
    Choose(sel(randperm(numel(sel),K))) = true;
else
    while sum(Choose) < K
        unSel = find(~Choose);
        [~,x] = min(max(Cosine(~Choose,Choose),[],2));
        Choose(unSel(x)) = true;
    end
end
end
