classdef OS_AE_GAM < ALGORITHM
% <2025> <multi/many> <real/integer/label/binary/permutation>
% OS_AE_GAM : Objective-space grouping + AE (PCA) generation + GAM back-mapping
%  - 目标空间分组：优先调用 OS_GroupByObjective(Population,'K',K_eff)
%    若未提供该文件，则使用内置谱聚类（含小M/NaN防护与K钳制）
%  - 组内 AE：用 PCA 作为线性自编码器，在潜空间做一轮 DE 采样生成新的目标子向量
%  - GAM 回写：对每组的写回变量子集用 GAM（fitrgam）建模；若不可用则回退岭回归（含加性多项式基）
%
% 运行示例：
%   main('-algorithm',@OS_AE_GAM,'-problem',@DTLZ2,'-M',3,'-D',12,'-N',92,'-evaluation',5e4);

    methods
        function main(obj, Problem)
            % 参数（可在命令行覆盖）
            %   K        : 目标空间簇数（自动钳制到 < M）
            %   topFrac  : 每组写回的变量占比（唯一分配后取前 topFrac）
            %   lambda   : 岭回归/拟合T与回退岭的正则
            %   eta      : 决策写回步长
            %   kcap     : AE潜空间最大维度
            %   kfrac    : AE潜空间维度占比
            %   F,Cr     : 潜空间 DE 参数
            %   period   : 每隔多少代重做一次目标分组与指派
            [K,topFrac,lambda,eta,kcap,kfrac,F,Cr,period] = ...
                obj.ParameterSet(3,0.2,1e-5,0.5,4,1/3,0.7,0.9,5);

            % 初始化
            Population = Problem.Initialization();
            gen = 1;

            % 初次：目标空间分组 + 写回变量指派 + 拟合 T
            [O_groups, S_groups, T] = group_and_assign(Problem, Population, K, topFrac, lambda);

            while obj.NotTerminated(Population)
                % 周期性更新分组与指派
                if gen == 1 || mod(gen-1,period)==0
                    [O_groups, S_groups, T] = group_and_assign(Problem, Population, K, topFrac, lambda);
                end

                X  = Population.decs;   % N×D
                Y  = Population.objs;   % N×M
                OffDec = X;

                % —— 逐目标组：AE(PCA) 生成 → GAM/Ridge 回写各自的 Sk
                for kgrp = 1:numel(O_groups)
                    Om = O_groups{kgrp};      % 该组的目标索引
                    Sk = S_groups{kgrp};      % 建议写回的决策变量索引
                    if isempty(Om) || isempty(Sk), continue; end

                    Yk = Y(:,Om);                         % N×|Om|
                    % AE: PCA 作为线性自编码器
                    [Ykz, muY, sgY] = safe_zscore(Yk);
                    r = max(1, min([size(Yk,2), max(1, ceil(size(Yk,2)*kfrac)), kcap]));
                    Wk = pca_basis(Ykz, r);               % |Om|×r
                    Z  = Ykz * Wk;                        % N×r
                    Znew = latent_DE(Z, F, Cr);           % 一轮 DE
                    Yk_new = (Znew * Wk') .* sgY + muY;   % 组内新目标子向量

                    % 用 GAM/Ridge 将 Y(:,Om) → x_j（对每个 j∈Sk 单独模型）
                    for jj = 1:numel(Sk)
                        j = Sk(jj);
                        mdl = fit_gam_or_ridge(Yk, X(:,j), lambda);
                        xj_hat = predict_gam_or_ridge(mdl, Yk_new);
                        OffDec(:,j) = (1-eta)*OffDec(:,j) + eta*xj_hat;
                    end
                end

                % 可行域裁剪 + 评估 + 环境选择
                OffDec    = min(max(OffDec, Problem.lower), Problem.upper);
                Offspring = Problem.Evaluation(OffDec);
                Population = env_select_local([Population, Offspring], Problem.N);

                gen = gen + 1;
            end
        end
    end
end

%% ========================= 内联工具（含修复） =========================

function [O_groups, S_groups, T] = group_and_assign(Problem, Population, K, topFrac, lambda)
% 目标空间分组 + 用 T 的能量唯一分配写回变量子集（含 K 钳制与小M防护）
X = Population.decs; Y = Population.objs;
[~,D] = size(X);  M = size(Y,2);

% —— K 钳制：确保 kmeans 样本数(=M) > 簇数
if isempty(K) || ~isscalar(K) || K < 1
    K = min(3, M);
end
if M == 1
    K_eff = 1;
else
    K_eff = min(K, max(1, M-1));     % 保证 M > K_eff
end

% 先尝试用户自带 OS_GroupByObjective；不存在则走内置
try
    [Og,~,~] = OS_GroupByObjective(Population,'K',K_eff);
    O_groups = Og;
catch
    O_groups = os_group_by_objective_builtin(Y, K_eff);
end

% 拟合 T（岭回归）：M×D
T = (Y.'*Y + lambda*eye(M)) \ (Y.'*X);

% —— 唯一分配：每个变量分给“对该变量影响能量最大”的目标组
Keff = numel(O_groups);
energy = zeros(Keff, D);
for k = 1:Keff
    Om = O_groups{k};
    if isempty(Om), continue; end
    for j = 1:D
        energy(k,j) = norm(T(Om,j), 2);   % 该变量在该目标组的耦合强度
    end
end
[~, owner] = max(energy, [], 1);

S_groups = cell(1,Keff);
for k = 1:Keff
    idx = find(owner==k);
    if isempty(idx), S_groups{k} = []; continue; end
    ek = energy(k, idx);
    [~,ord] = sort(ek, 'descend');
    topK = max(1, round(topFrac*numel(idx)));
    S_groups{k} = idx(ord(1:topK));
end
end

function O_groups = os_group_by_objective_builtin(Y, K)
% 仅用 Y 的谱聚类（基于非负 cosine 相似度），含 NaN 清理与 K 钳制
[~,M] = size(Y);

% K 钳制与退化处理
if M == 1 || K <= 1
    O_groups = {1:M};
    return;
end
K = min(K, max(1, M-1));   % 保证 M > K

% 标准化并清 NaN
Yz = zscore(Y,0,1);
Yz(:, any(isnan(Yz),1)) = 0;

% cosine 相似度（列即目标）
nrm = sqrt(sum(Yz.^2,1)); nrm(nrm==0) = 1;
U = Yz ./ nrm;
S = U.' * U;                   % M×M
S(1:M+1:end) = 0;

% 非负亲和矩阵 → 归一化拉普拉斯
A = max(S,0);  A = A - diag(diag(A));
d = sum(A,2);
Dmat = diag(d + eps);          % 防 0 度
Lsym = eye(M) - Dmat^(-1/2) * A * Dmat^(-1/2);
Lsym = (Lsym + Lsym.')/2;

% 特征分解 + 行归一
[V,E] = eig(Lsym);
[~,ord] = sort(diag(E),'ascend');
H  = V(:,ord(1:K));
Hn = H ./ max(sqrt(sum(H.^2,2)), eps);
Hn(~isfinite(Hn)) = 0;

% kmeans：样本 = 目标维 M（保证 M > K）
opts = statset('MaxIter',200,'Display','off');
repl = max(5, min(10, M-1));
lbl = kmeans(Hn, K, 'Replicates', repl, 'Options', opts);

% 组装
O_groups = cell(1,K);
for k = 1:K
    O_groups{k} = find(lbl==k).';
end
end

function mdl = fit_gam_or_ridge(Yin, x, lambda)
% 优先 fitrgam；失败则回退到“加性多项式基 + 岭回归”
try
    mdl0 = fitrgam(Yin, x, 'Interactions','none', 'Standardize',true);
    mdl.type = 'gam'; mdl.mdl = mdl0;
catch
    Phi = design_additive_poly(Yin, 3);   % N×(M*3)
    b = (Phi.'*Phi + lambda*eye(size(Phi,2))) \ (Phi.'*x);
    mdl.type = 'ridge'; mdl.mdl = b;
end
end

function xhat = predict_gam_or_ridge(mdl, Yin_new)
switch mdl.type
    case 'gam'
        xhat = predict(mdl.mdl, Yin_new);
    otherwise
        Phi_new = design_additive_poly(Yin_new, 3);
        xhat = Phi_new * mdl.mdl;
end
end

function Phi = design_additive_poly(Yin, deg)
% 无交互的加性多项式特征：每个目标维独立的 [y, y^2, ..., y^deg]
[N,M] = size(Yin);
Phi = zeros(N, M*deg);
col = 0;
for m = 1:M
    y = Yin(:,m);
    for d = 1:deg
        col = col + 1;
        Phi(:,col) = y.^d;
    end
end
end

function [Z,mu,sg] = safe_zscore(X)
mu = mean(X,1);
sg = std(X,0,1);
sg(sg==0) = 1;
Z = (X - mu) ./ sg;
end

function W = pca_basis(X, k)
% X 已标准化；返回前 k 个主方向（列正交）
[~,~,V] = svd(X,'econ');
W = V(:,1:k);
end

function Znew = latent_DE(Z, F, Cr)
% 潜空间做 DE/rand/1/bin
[N,k] = size(Z);
Znew = Z;
if N < 4 || k==0, return; end
idx = 1:N;
for i = 1:N
    r = idx; r(i) = [];
    if numel(r) < 3, Znew(i,:) = Z(i,:); continue; end
    r = r(randperm(numel(r),3));
    r1 = r(1); r2 = r(2); r3 = r(3);
    v = Z(r1,:) + F*(Z(r2,:) - Z(r3,:));
    jrand = randi(k);
    mask = (rand(1,k) < Cr); mask(jrand) = true;
    u = Z(i,:); u(mask) = v(mask);
    Znew(i,:) = u;
end
end

function Population = env_select_local(Population, N)
% 非支配排序 + 角度截断（轻量 ES）
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
% 归一化 + 基于 cosine 的多样性截断
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
