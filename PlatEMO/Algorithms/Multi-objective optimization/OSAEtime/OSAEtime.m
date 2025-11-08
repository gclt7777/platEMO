classdef OSAEtime < ALGORITHM
% <2025> <multi/many> <real/integer/label/binary/permutation> <large>
% OSAEtime : Objective-space grouping + AE(PCA) + Linear T write-back with time-controlled updates
%
% - 目标聚类 → 唯一指派变量列 S_k（按 T 子块能量）
% - 组内：PCA 潜空间 + DE(rand/1/bin) 生成新的目标向量
% - 写回：仅改各自 S_k 列，半步融合 eta
% - T 为 zscore 域的岭回归 SVD 解；支持 EMA 平滑
% - periodGroup：重分组/重指派周期；tPeriod：T 的拟合周期（独立于分组）

    methods
        function main(obj, Problem)
            %% 参数
            %  K,topFrac,lambda,eta,kcap,kfrac,F,Cr,periodGroup,emaA,tPeriod
            [K, topFrac, lambda, eta, kcap, kfrac, F, Cr, periodGroup, emaA, tPeriod] = ...
                obj.ParameterSet(3, 0.2, 1e-5, 0.5, 4, 1/3, 0.7, 0.9, 5, 0.0, 1);

            %% 初始化
            Population = Problem.Initialization();
            gen      = 1;
            map      = [];
            map_prev = [];

            D = Problem.D;
            M = Problem.M;

            % 参考点（NSGA-III）——生成一次全程复用
            Z = make_reference_points(Problem.N, M);

            % 首代：分组 + 拟合 T + 指派
            O_groups  = group_by_objective(Population, K);
            map       = fitMap_ridgeSVD(Population.objs, Population.decs, lambda);
            if emaA > 0
                map_prev = map;
            end
            S_groups  = assign_S_groups(map, O_groups, topFrac, D);

            %% 主循环
            while obj.NotTerminated(Population)
                X = Population.decs;   % N×D
                Y = Population.objs;   % N×M
                OffDec = X;

                % —— 节奏控制：分组 与 T 拟合（互相独立） ——
                doReGroup = (gen == 1) || (mod(gen-1, periodGroup) == 0);
                doFitT    = (gen == 1) || (mod(gen-1, tPeriod) == 0);

                % —— 分组（仅按 Y），不动 T ——
                if doReGroup
                    O_groups = group_by_objective(Population, K);
                end

                % —— 拟合/更新 T（可 EMA 平滑） ——
                if doFitT
                    map_new = fitMap_ridgeSVD(Y, X, lambda);
                    if emaA > 0 && ~isempty(map_prev)
                        map_blend.T   = (1-emaA)*map_prev.T + emaA*map_new.T;
                        map_blend.muY = map_new.muY;  map_blend.sigY = map_new.sigY;
                        map_blend.muX = map_new.muX;  map_blend.sigX = map_new.sigX;
                        map = map_blend;
                        map_prev = map_blend;
                    else
                        map = map_new;
                        map_prev = map_new;
                    end
                end

                % —— 若分组或 T 变了，则重算唯一指派 S_groups ——
                if doReGroup || doFitT || isempty(S_groups)
                    S_groups = assign_S_groups(map, O_groups, topFrac, D);
                end

                % —— 分组：PCA→DE 生成，并用子块 T(Ok,Sk) 写回各自列 ——
                for k = 1:numel(O_groups)
                    Ok = O_groups{k};  Sk = S_groups{k};
                    if isempty(Ok) || isempty(Sk)
                        continue;
                    end

                    Yk      = Y(:, Ok);
                    Yk_new  = ae_pca_generate(Yk, kcap, kfrac, F, Cr);      % N×|Ok|
                    Xhat_Sk = apply_map_sub(map, Yk_new, Ok, Sk);           % 反标准化写回

                    % 半步融合
                    OffDec(:, Sk) = (1-eta)*OffDec(:, Sk) + eta*Xhat_Sk;
                end

                % 边界裁剪 + 评估 + 环境选择（NSGA-III）
                OffDec    = min(max(OffDec, Problem.lower), Problem.upper);
                Offspring = Problem.Evaluation(OffDec);
                Population = env_select_nsga3([Population, Offspring], Problem.N, Z);

                gen = gen + 1;
            end
        end
    end
end

%% ============ 目标分组 / 唯一指派 ============

function O_groups = group_by_objective(Population, K)
Y = Population.objs; M = size(Y,2);
if isempty(K) || ~isscalar(K) || K < 1, K = min(3, M); end
if M == 1
    K_eff = 1;
else
    K_eff = min(K, max(1, M-1));
end
try
    [Og, ~, ~] = OS_GroupByObjective(Population, 'K', K_eff);
    O_groups = Og;
catch
    O_groups = os_group_by_objective_builtin(Y, K_eff);
end
end

function S_groups = assign_S_groups(map, O_groups, topFrac, D)
% 稳健：topFrac∈[0,1]；输出统一为列向量（空为 0×1）
topFracEff = min(max(topFrac, 0), 1);
Keff = numel(O_groups);
energy = zeros(Keff, D);
for k = 1:Keff
    Ok = O_groups{k};
    if isempty(Ok)
        continue;
    end
    Tk = map.T(Ok, :);                    % |Ok|×D
    energy(k, :) = sqrt(sum(Tk.^2, 1));   % 子块能量
end
[~, owner] = max(energy, [], 1);          % 每列归属组

S_groups = cell(1, Keff);
for k = 1:Keff
    idx = find(owner == k);
    cnt = numel(idx);
    if cnt == 0
        S_groups{k} = zeros(0,1);
        continue;
    end
    e = energy(k, idx); e(~isfinite(e)) = -inf;
    [~, ord] = sort(e, 'descend');
    topK = max(1, min(cnt, round(topFracEff*cnt)));
    S_groups{k} = reshape(idx(ord(1:topK)), [], 1); % 强制列向量
end
end

%% ============ 拟合/生成 ============

function map = fitMap_ridgeSVD(Y, X, lambda)
% 在标准化域解 T，并保存反标准化所需统计量
muY = mean(Y,1);  sigY = std(Y,0,1); sigY(sigY==0) = 1;
muX = mean(X,1);  sigX = std(X,0,1); sigX(sigX==0) = 1;

Yz = (Y - muY)./sigY;
[U,S,V] = svd(Yz,'econ'); sig = diag(S);
G  = V * diag(sig./(sig.^2 + lambda)) * U';   % (Y'Y+λI)^{-1}Y'
Xz = (X - muX)./sigX;
T  = G * Xz;                                   % M×D

map.T = T; map.muY = muY; map.sigY = sigY; map.muX = muX; map.sigX = sigX;
end

function Y_new = ae_pca_generate(Yk, kcap, kfrac, F, Cr)
[Ykz, mu, sg] = safe_zscore(Yk);
r  = max(1, min([size(Yk,2), max(1, ceil(size(Yk,2)*kfrac)), kcap]));
W  = pca_basis(Ykz, r);
Z  = Ykz * W;

% 一步 DE(rand/1/bin) 于潜空间
[N, k] = size(Z); Znew = Z;
if N >= 4 && k > 0
    idx = 1:N;
    for i = 1:N
        rset = idx; rset(i) = [];
        rset = rset(randperm(numel(rset), 3));
        v = Z(rset(1), :) + F*(Z(rset(2), :) - Z(rset(3), :));
        jrand = randi(k);
        mask = (rand(1, k) < Cr); mask(jrand) = true;
        u = Z(i, :); u(mask) = v(mask);
        Znew(i, :) = u;
    end
end
Y_new = (Znew * W') .* sg + mu;
end

function [Z, mu, sg] = safe_zscore(X)
mu = mean(X,1);
sg = std(X,0,1); sg(sg==0) = 1;
Z  = (X - mu) ./ sg;
end

function W = pca_basis(X, k)
[~,~,V] = svd(X,'econ'); W = V(:,1:k);
end

function Xhat_Sk = apply_map_sub(map, Yk_new, Ok, Sk)
Yz  = bsxfun(@rdivide, bsxfun(@minus, Yk_new, map.muY(Ok)), map.sigY(Ok));
Xz  = Yz * map.T(Ok, Sk);
Xhat_Sk = bsxfun(@plus, bsxfun(@times, Xz, map.sigX(Sk)), map.muX(Sk));
end

%% ============ 环境选择：NSGA-III ============

function Population = env_select_nsga3(PopBoth, N, Z)
PopObj = PopBoth.objs;

% 非支配排序
[FrontNo, MaxFNo] = NDSort(PopObj, N);
Next    = FrontNo < MaxFNo;

% —— 统一成列向量，避免 vertcat 维度不一致 ——
PopKeep = find(Next(:));              % 已保留（列向量）
Last    = find(FrontNo == MaxFNo);    % 最后一层（列向量）

K = N - numel(PopKeep);
if K <= 0
    Select = PopKeep(:);
    Select = Select(1:min(numel(Select), N));
    Population = PopBoth(Select);
    return;
end

% 若最后层候选不足，先压到上限；后面再补齐
K = min(K, numel(Last));

% 归一化（ASF 截距；失败退回 min–max）
[PopNorm, ~, ~] = normalize_for_nsga3(PopObj);
LastNorm = PopNorm(Last, :);

% 参考点归属
AssocAll = associate_to_refs(PopNorm, Z);                % 所有个体的归属（列）
rho      = accumarray(AssocAll(PopKeep), 1, [size(Z,1), 1]);

[AssocLast, PerpLast] = associate_to_refs(LastNorm, Z); % 最后一层的归属与垂距（列）

Chosen = false(numel(Last), 1);                          % 逻辑列向量
picked = 0;

while picked < K
    minRho = min(rho);
    candZ  = find(rho == minRho);                        % 这些参考点负载最小
    zPick  = candZ(randi(numel(candZ)));

    idx = find(~Chosen & (AssocLast == zPick));          % 属于该参考点且未选
    if isempty(idx)
        % 该参考点无候选 → 在所有未选里随机选一个兜底
        idx = find(~Chosen);
        if isempty(idx), break; end
        pick = idx(randi(numel(idx)));
    else
        if rho(zPick) == 0
            % niche 为空 → 选垂距最小
            [~, t] = min(PerpLast(idx));
            pick = idx(t);
        else
            % niche 已有 → 随机选
            pick = idx(randi(numel(idx)));
        end
    end
    Chosen(pick) = true;
    rho(zPick)   = rho(zPick) + 1;
    picked       = picked + 1;
end

% 组合索引（全部转列再拼）
cands  = Last(Chosen);
Select = [PopKeep(:); cands(:)];

% 若还不够 N（极端情况），从剩余 Last 里补；若超了，截断
if numel(Select) < N
    rest = setdiff(Last(:), cands(:), 'stable');
    need = min(N - numel(Select), numel(rest));
    if need > 0
        Select = [Select; rest(1:need)];
    end
elseif numel(Select) > N
    Select = Select(1:N);
end

Population = PopBoth(Select);
end


function [PopNorm, zmin, intercepts] = normalize_for_nsga3(PopObj)
% 理想点
zmin = min(PopObj, [], 1);
PopShift = bsxfun(@minus, PopObj, zmin);

% ASF 极点（每一维单独最小化）求近似极端点
M = size(PopObj,2);
W = eye(M); W(W==0) = 1e-6;
asf = @(f,w) max(bsxfun(@rdivide, f, w), [], 2);
ExtremeIdx = zeros(1,M);
for m = 1:M
    [~, ExtremeIdx(m)] = min(asf(PopShift, W(m,:)));
end
ExtremePts = PopShift(ExtremeIdx, :);

% 拟合超平面求截距
intercepts = ones(1,M);
try
    A = ExtremePts;
    b = ones(M,1);
    alpha = A\b;                      % 解权重
    intercepts = 1./alpha(:)';        % 截距
    % 若有非正或NaN，视为退化
    if any(~isfinite(intercepts) | intercepts<=0)
        error('degenerate');
    end
catch
    % 退化：改用 min–max 归一化
    fmax = max(PopObj, [], 1);
    span = fmax - zmin;
    span(span==0) = 1;
    PopNorm = bsxfun(@rdivide, bsxfun(@minus, PopObj, zmin), span);
    return;
end

% 正常：按截距缩放
PopNorm = bsxfun(@rdivide, PopShift, intercepts);
PopNorm(~isfinite(PopNorm)) = 0;
end

function [assoc, perp] = associate_to_refs(F, Z)
% 余弦 + 垂距
% 先把参考点单位化
Zn = Z ./ max(sqrt(sum(Z.^2,2)), eps);
Fn = F; nrm = sqrt(sum(Fn.^2,2)); nrm(nrm==0)=1; Fn = Fn ./ nrm;

cosine = Fn * Zn.';                             % N×K
cosine = max(min(cosine,1),-1);
[~, assoc] = max(cosine, [], 2);
% 垂直距离 = ||f|| * sqrt(1 - cos^2)
perp = sqrt(max(0, 1 - cosine.^2));             % 使用单位范数后的等价形式
row = (1:size(F,1))';
perp = perp(sub2ind(size(perp), row, assoc));
end

function Z = make_reference_points(N, M)
% 优先调用 PlatEMO 的 UniformPoint；不可用则内置
try
    [Z, ~] = UniformPoint(N, M);
    if size(Z,1) < N
        % 不足则重复拼接并截断
        rep = ceil(N/size(Z,1));
        Z = repmat(Z, rep, 1);
        Z = Z(1:N, :);
    elseif size(Z,1) > N
        Z = Z(1:N, :);
    end
catch
    Z = uniform_points_builtin(N, M);
end
% 单位化（防数值问题）
Z = Z ./ max(sum(Z,2), eps);
end

function Z = uniform_points_builtin(N, M)
% 简化 Das-Dennis：优先单层 H1，不足再加一层 H2
H1 = 1;
while nchoosek(H1+M-1, M-1) < N && H1 < 20
    H1 = H1 + 1;
end
Z = das_dennis(H1, M);
if size(Z,1) < N
    H2 = 1;
    Z2 = das_dennis(H2, M)/2 + 1/(2*M);
    Z = [Z; Z2];
    while size(Z,1) < N && H2 < 20
        H2 = H2 + 1;
        Z2 = das_dennis(H2, M)/2 + 1/(2*M);
        Z = [Z; Z2];
    end
end
if size(Z,1) > N
    Z = Z(1:N, :);
end
end

function W = das_dennis(H, M)
% 生成和为1的等距格点
% 递归实现
W = [];
a = zeros(1,M);
recur(1, 1);
    function recur(cur, left)
        if cur == M
            a(cur) = left;
            W = [W; a/H];
        else
            for i = 0:left
                a(cur) = i;
                recur(cur+1, left-i);
            end
        end
    end
end

%% ============ 内置的目标聚类兜底 ============

function O_groups = os_group_by_objective_builtin(Y, K)
[~, M] = size(Y);
if M == 1 || K <= 1, O_groups = {1:M}; return; end
K = min(K, max(1, M-1));

Yz = zscore(Y, 0, 1);
Yz(~isfinite(Yz)) = 0;               % 轻量防护
nrm = sqrt(sum(Yz.^2, 1)); nrm(nrm==0) = 1;
U = Yz ./ nrm;
S = U.'*U; S(1:M+1:end) = 0;

A = max(S, 0); A = A - diag(diag(A));
d = sum(A, 2); Dm = diag(d + eps);
Lsym = eye(M) - Dm^(-1/2)*A*Dm^(-1/2); Lsym = (Lsym+Lsym.')/2;

[V,E] = eig(Lsym);
[~, ord] = sort(diag(E), 'ascend');
H  = V(:, ord(1:K));
Hn = H ./ max(sqrt(sum(H.^2, 2)), eps); Hn(~isfinite(Hn)) = 0;

try
    opts = statset('MaxIter', 200, 'Display', 'off');
    repl = max(5, min(10, M-1));
    lbl  = kmeans(Hn, K, 'Replicates', repl, 'Options', opts);
catch
    % 无统计工具箱时兜底：按顺序均匀分配
    idx = 1:M; lbl = mod(idx-1, K) + 1;
end
O_groups = cell(1, K);
for k = 1:K
    O_groups{k} = find(lbl == k).';
end
end
