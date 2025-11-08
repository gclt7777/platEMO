classdef OSAE_RCD < ALGORITHM
% <2025> <multi/many> <real/integer/label/binary/permutation> <large>
% OSAE_RCD : Row–Column Dual-Drive for O–S–D
% 列相位 C-phase（组内强收敛）：目标分组 Ok → 拟合/更新 T → AE(PCA) 组内生成 → 仅在 Sk 列写回
% 行相位 D-phase（全局强多样）：
%   (1) ACP 锚点凸组合：把个体朝“全局稀疏方向锚点”小幅拉动目标，再经组块 T 写回
%   (2) NSE 近零敏方向探索：在 Sk 列沿 T(Ok,Sk) 的“弱奇异方向”微扰，产生不同决策模态
% 环境选择：非支配 + 角度截断
%
% 参数（可命令行覆盖）
%   K           —— 目标/决策组数（默认 3，内部钳制 ≤ M-1）
%   lambda      —— 岭回归系数（默认 1e-5）
%   eta         —— 半步融合系数（默认 0.5）
%   kcap        —— 组内 PCA 潜空间维上限（默认 4）
%   kfrac       —— 组内 PCA 潜空间维占比（默认 1/3）
%   F, Cr       —— 潜空间 DE(rand/1/bin) 参数（默认 0.7, 0.9）
%   periodGroup —— 分组周期（默认 5）
%   tPeriod     —— T 拟合周期（默认 1）
%   topFrac     —— 指派到 Sk 时每组取前比例（默认 0.2）
%   gammaMax    —— D-phase 注入最大占比（默认 0.30）
%   alphaMix    —— ACP 锚点凸组合强度（默认 0.10）
%   epsNSE      —— NSE 相对步长（默认 0.05）
%   anchorPeriod—— 锚点更新周期（默认 5）
%   kAnchors    —— 锚点数量（默认 = max(3, min(2*M, 24)))
%
% 示例：
%   main('-algorithm',@OSAE_RCD,'-problem',@DTLZ3,'-M',10,'-D',500,'-N',276,'-evaluation',1e5)
%
    methods
        function main(obj, Problem)
            %% 参数
            [K, lambda, eta, kcap, kfrac, F, Cr, periodGroup, tPeriod, topFrac, ...
             gammaMax, alphaMix, epsNSE, anchorPeriod, kAnchors] = ...
             obj.ParameterSet(3, 1e-5, 0.5, 4, 1/3, 0.7, 0.9, 5, 1, 0.2, ...
                               0.30, 0.10, 0.05, 5, 0);

            %% 初始化
            Population = Problem.Initialization();
            gen = 1; N = Problem.N; D = Problem.D; M = size(Population.objs,2);
            if kAnchors<=0, kAnchors = max(3, min(2*M, 24)); end

            % 统计与映射
            [~, muY, sigY] = safe_zscore(Population.objs);
            [~, muX, sigX] = safe_zscore(Population.decs);
            map.muY = muY; map.sigY = sigY; map.muX = muX; map.sigX = sigX; map.T = zeros(M, D);

            % 初代：分组 + 拟合 T + 指派
            O_groups  = group_by_objective(Population, K);
            map       = fitMap_ridgeSVD(Population.objs, Population.decs, lambda);
            S_groups  = assign_S_groups(map, O_groups, topFrac, D);

            % 锚点缓存
            anchors = compute_anchors(Population.objs, kAnchors);

            %% 主循环
            while obj.NotTerminated(Population)
                X = Population.decs;   % N×D
                Y = Population.objs;   % N×M
                OffDec = X;

                % ===== 列相位 C-phase：组内 AE → 写回 =====
                doReGroup = (gen == 1) || (mod(gen-1, periodGroup) == 0);
                doFitT    = (gen == 1) || (mod(gen-1, tPeriod) == 0);
                if doReGroup
                    O_groups = group_by_objective(Population, K);
                end
                if doFitT
                    map = fitMap_ridgeSVD(Y, X, lambda);
                end
                if doReGroup || doFitT || isempty(S_groups)
                    S_groups = assign_S_groups(map, O_groups, topFrac, D);
                end

                for k = 1:numel(O_groups)
                    Ok = O_groups{k};  Sk = S_groups{k};
                    if isempty(Ok) || isempty(Sk), continue; end
                    Yk      = Y(:, Ok);
                    Yk_new  = ae_pca_generate(Yk, kcap, kfrac, F, Cr);
                    Xhat_Sk = apply_map_sub(map, Yk_new, Ok, Sk);
                    OffDec(:, Sk) = (1-eta)*OffDec(:, Sk) + eta*Xhat_Sk;
                end

                % ===== 行相位 D-phase：ACP + NSE 注入（小占比） =====
                if mod(gen-1, anchorPeriod) == 0
                    anchors = compute_anchors(Population.objs, kAnchors);
                end
                nInject = max(1, floor(gammaMax * N));
                idxInject = select_diverse_rows(Y, nInject); % 贪心最小角/最远点近似

                % —— ACP：向锚点凸组合 ——
                OffDecD_rows = OffDec(idxInject, :);
                for t = 1:nInject
                    i = idxInject(t);
                    y = Y(i, :);
                    a = pick_farthest_anchor(y, anchors);
                    y_mix = (1 - alphaMix)*y + alphaMix*a;
                    % 分组写回
                    x_row = OffDecD_rows(t, :);
                    for k = 1:numel(O_groups)
                        Ok = O_groups{k}; Sk = S_groups{k};
                        if isempty(Ok) || isempty(Sk), continue; end
                        yk_mix = y_mix(:, Ok);
                        xhat   = apply_map_sub(map, yk_mix, Ok, Sk);
                        x_row(:, Sk) = (1-eta)*x_row(:, Sk) + eta*xhat;
                    end
                    OffDecD_rows(t, :) = x_row;
                end

                % —— NSE：沿弱奇异方向微扰 ——
                OffDecD_rows = nse_perturb_rows(OffDecD_rows, O_groups, S_groups, map, epsNSE, Problem.lower, Problem.upper);

                % ===== 评估与选择 =====
                OffDec    = min(max(OffDec, Problem.lower), Problem.upper);
                OffspringC = Problem.Evaluation(OffDec);

                OffDecD_rows = min(max(OffDecD_rows, Problem.lower), Problem.upper);
                OffspringD = Problem.Evaluation(OffDecD_rows);

                Population = env_select_nsgaiii([Population, OffspringC, OffspringD], Problem.N)

                gen = gen + 1;
            end
        end
    end
end

%% ===================== 分组 / 指派 / 映射 =====================
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
% 以子块能量进行唯一指派：每列归属能量最大的组，再在组内取前 topFrac
Keff  = numel(O_groups);
energy = zeros(Keff, D);
for k = 1:Keff
    Ok = O_groups{k};
    if isempty(Ok), continue; end
    Tk = map.T(Ok, :);                    % |Ok|×D
    energy(k, :) = sqrt(sum(Tk.^2, 1));   % 列能量
end
[~, owner] = max(energy, [], 1);
S_groups = cell(1, Keff);
frac = min(max(topFrac, 0), 1);
for k = 1:Keff
    idx = find(owner == k); cnt = numel(idx);
    if cnt == 0, S_groups{k} = zeros(0,1); continue; end
    e = energy(k, idx); e(~isfinite(e)) = -inf;
    [~, ord] = sort(e, 'descend'); topK = max(1, min(cnt, floor(frac*cnt)));
    S_groups{k} = reshape(idx(ord(1:topK)), [], 1);
end
end

function map = fitMap_ridgeSVD(Y, X, lambda)
% 在标准化域解 T，并保存反标准化统计
muY = mean(Y,1);  sigY = std(Y,0,1); sigY(sigY==0) = 1;
muX = mean(X,1);  sigX = std(X,0,1); sigX(sigX==0) = 1;
Yz = (Y - muY)./sigY; [U,S,V] = svd(Yz,'econ'); sig = diag(S);
G  = V * diag(sig./(sig.^2 + lambda)) * U';   % (Y'Y+λI)^{-1}Y'
Xz = (X - muX)./sigX; T  = G * Xz;            % M×D
map.T = T; map.muY = muY; map.sigY = sigY; map.muX = muX; map.sigX = sigX;
end

%% ===================== 组内生成 / 写回 / 选择 =====================
function Y_new = ae_pca_generate(Yk, kcap, kfrac, F, Cr)
[Ykz, mu, sg] = safe_zscore(Yk);
r  = max(1, min([size(Yk,2), max(1, ceil(size(Yk,2)*kfrac)), kcap]));
W  = pca_basis(Ykz, r);
Z  = Ykz * W; [N,k] = size(Z); Znew = Z;
if N >= 4 && k > 0
    idx = 1:N;
    for i = 1:N
        rset = idx; rset(i) = [];
        rset = rset(randperm(numel(rset), 3));
        v = Z(rset(1), :) + F*(Z(rset(2), :) - Z(rset(3), :));
        jrand = randi(k); mask = (rand(1, k) < Cr); mask(jrand) = true;
        u = Z(i, :); u(mask) = v(mask); Znew(i, :) = u;
    end
end
Y_new = (Znew * W') .* sg + mu;
end

function Xhat_Sk = apply_map_sub(map, Yk_new, Ok, Sk)
Yz  = bsxfun(@rdivide, bsxfun(@minus, Yk_new, map.muY(Ok)), map.sigY(Ok));
Xz  = Yz * map.T(Ok, Sk);
Xhat_Sk = bsxfun(@plus, bsxfun(@times, Xz, map.sigX(Sk)), map.muX(Sk));
end

function Population = env_select_local(PopBoth, N)
PopObj = PopBoth.objs; [FrontNo, MaxFNo] = NDSort(PopObj, N);
Next = FrontNo < MaxFNo; Last = find(FrontNo == MaxFNo); K = N - sum(Next);
if K > 0
    Choose = truncation_angle(PopObj(Last, :), K); Next(Last(Choose)) = true;
end
Population = PopBoth(Next);
end

function Choose = truncation_angle(PopObj, K)
fmax = max(PopObj, [], 1); fmin = min(PopObj, [], 1); span = fmax - fmin; span(span==0) = 1;
P = (PopObj - fmin) ./ span; nrm = sqrt(sum(P.^2, 2)); nrm(nrm==0) = 1; U = P ./ nrm;
Cosine = U*U.'; Cosine(1:size(Cosine,1)+1:end) = 0;
Choose = false(1, size(P,1)); [~, extreme] = max(P, [], 1); Choose(extreme) = true;
if sum(Choose) > K
    sel = find(Choose); Choose = false(1, size(P,1)); Choose(sel(randperm(numel(sel), K))) = true;
else
    while sum(Choose) < K
        unSel = find(~Choose);
        [~, x] = min(max(Cosine(~Choose, Choose), [], 2));
        Choose(unSel(x)) = true;
    end
end
end

%% ===================== 行相位：锚点与注入 =====================
function anchors = compute_anchors(Y, kAnchors)
% 在归一化目标空间做贪心远点采样（近似 DPP），返回锚点集合及归一化统计
[Yh, fmin, fmax] = normalize_obj(Y);
numPts = size(Y,1);
if numPts == 0
    anchors.values = zeros(0, size(Y,2));
    anchors.normalized = anchors.values;
    anchors.fmin = fmin;
    anchors.fmax = fmax;
    return;
end
kEff = min(max(1, floor(kAnchors)), numPts);
% 先选与均值最远的点，再做最远点采样
mu = mean(Yh,1);
[~, i0] = max(sum((Yh - mu).^2, 2));
sel = i0;
cosCache = clamp_cos(Yh * Yh(sel, :).');
while numel(sel) < kEff
    dmin = min(1 - cosCache, [], 2);
    dmin(sel) = -inf;
    [~, ix] = max(dmin);
    if any(sel == ix)
        break;
    end
    sel(end+1) = ix; %#ok<AGROW>
    cosCache = [cosCache, clamp_cos(Yh * Yh(ix, :).')]; %#ok<AGROW>
end
sel = unique(sel, 'stable');
anchors.values = Y(sel, :);
anchors.normalized = Yh(sel, :);
anchors.fmin = fmin;
anchors.fmax = fmax;
end

function idx = select_diverse_rows(Y, n)
% 贪心 max-min 选择 n 个多样行（用角距离近似）
[Yh, ~, ~] = normalize_obj(Y); N = size(Yh,1);
if n >= N, idx = 1:N; return; end
mu = mean(Yh,1); [~, i0] = max(sum((Yh - mu).^2, 2));
sel = i0; cand = true(N,1); cand(i0)=false;
cosCache = clamp_cos(Yh * Yh(sel, :).');
while numel(sel) < n
    dmin = min(1 - cosCache, [], 2);
    dmin(~cand) = -inf;
    [~, ix] = max(dmin);
    sel(end+1) = ix; cand(ix) = false;
    cosCache = [cosCache, clamp_cos(Yh * Yh(ix, :).')]; %#ok<AGROW>
end
idx = sel;
end

function a = pick_farthest_anchor(y, anchors)
if isempty(anchors.values)
    a = y;
    return;
end
yhat = normalize_obj(y, anchors.fmin, anchors.fmax);
cosv = anchors.normalized * yhat';
[~, ix] = min(cosv);
a = anchors.values(ix, :);
end

function [Yh, fmin, fmax] = normalize_obj(Y, fmin, fmax)
if isempty(Y)
    Yh = zeros(size(Y));
    if nargin < 2 || isempty(fmin) || isempty(fmax)
        fmin = zeros(1, size(Y,2));
        fmax = fmin;
    end
    return;
end
if nargin < 2 || isempty(fmin) || isempty(fmax)
    fmax = max(Y, [], 1);
    fmin = min(Y, [], 1);
end
span = fmax - fmin; span(span==0) = 1; Yn = (Y - fmin) ./ span;
Yn(~isfinite(Yn)) = 0;
nrm = sqrt(sum(Yn.^2, 2)); nrm(nrm==0) = 1; Yh = Yn ./ nrm;
end

function C = clamp_cos(C)
C = max(min(C, 1), -1);
end

function OffRows = nse_perturb_rows(OffRows, O_groups, S_groups, map, epsNSE, lower, upper)
% 沿 T 的弱奇异方向（小奇异值对应的右奇异向量）对 Sk 列微扰
if isempty(OffRows), return; end
for k = 1:numel(O_groups)
    Ok = O_groups{k}; Sk = S_groups{k}; if isempty(Ok) || isempty(Sk), continue; end
    Tk = map.T(Ok, Sk); if isempty(Tk), continue; end
    [~,S,V] = svd(Tk,'econ'); s = diag(S); if isempty(s), continue; end
    % 选“弱奇异方向”基：奇异值低于中位数的列；若全高，则取最后一列
    med = median(s); keep = find(s <= med);
    if isempty(keep), keep = size(V,2); end
    Vperp = V(:, keep); % |Sk|×q
    if isempty(Vperp), continue; end
    span = (upper(Sk) - lower(Sk)); span(span==0) = 1;
    step = epsNSE .* span(:)';
    Z = randn(size(OffRows,1), size(Vperp,2));
    delta = (Z * Vperp') .* step; % 行×|Sk|
    OffRows(:, Sk) = OffRows(:, Sk) + delta;
end
% 边界裁剪
OffRows = min(max(OffRows, lower), upper);
end

%% ===================== 工具函数 =====================
function [Z, mu, sg] = safe_zscore(X)
mu = mean(X,1); sg = std(X,0,1); sg(sg==0) = 1; Z = (X - mu) ./ sg;
end

function W = pca_basis(X, k)
[~,~,V] = svd(X,'econ'); W = V(:,1:k);
end

function O_groups = os_group_by_objective_builtin(Y, K)
[~, M] = size(Y);
if M == 1 || K <= 1, O_groups = {1:M}; return; end
K = min(K, max(1, M-1));
Yz = zscore(Y, 0, 1); Yz(~isfinite(Yz)) = 0; nrm = sqrt(sum(Yz.^2, 1)); nrm(nrm==0) = 1; U = Yz ./ nrm;
S = U.'*U; S(1:M+1:end) = 0; A = max(S, 0); A = A - diag(diag(A)); d = sum(A, 2); Dm = diag(d + eps);
Lsym = eye(M) - Dm^(-1/2)*A*Dm^(-1/2); Lsym = (Lsym+Lsym.')/2;
[V,E] = eig(Lsym); [~, ord] = sort(diag(E), 'ascend'); H  = V(:, ord(1:K));
Hn = H ./ max(sqrt(sum(H.^2, 2)), eps); Hn(~isfinite(Hn)) = 0;
try
    opts = statset('MaxIter', 200, 'Display', 'off'); repl = max(5, min(10, M-1));
    lbl  = kmeans(Hn, K, 'Replicates', repl, 'Options', opts);
catch
    idx = 1:M; lbl = mod(idx-1, K) + 1;
end
O_groups = cell(1, K); for k = 1:K, O_groups{k} = find(lbl == k).'; end
end

%% ===================== NSGA-III 环境选择（参考向量配额） =====================
function Population = env_select_nsgaiii(PopBoth, N)
PopObj = PopBoth.objs; [~,M] = size(PopObj);

% 1) 非支配排序
[FrontNo, MaxFNo] = NDSort(PopObj, N);
Next = FrontNo < MaxFNo;           % 直接进入的整层（逻辑向量/列）
Last = find(FrontNo == MaxFNo);    % 末层候选（列向量）
Kneed = N - sum(Next);             % 末层还需补的人数

if Kneed <= 0 || isempty(Last)
    Population = PopBoth(Next);
    return;
end

% 2) 生成参考向量（单位单纯形）
[V,~] = UniformPoint(N, M);
nv = sqrt(sum(V.^2,2)); nv(nv==0) = 1;
V = V ./ nv;

% 3) 归一化（在“已选整层 ∪ 末层候选”上做）
idxNext = find(Next);
if isempty(idxNext), idxNext = zeros(0,1); else, idxNext = idxNext(:); end
idxLast = Last(:);
ChooseIdx = [idxNext; idxLast];                 % ✅ 现在两边都是列向量，且空为 0×1
Fchoose   = PopObj(ChooseIdx, :);
[~, FN]   = normalize_nsga3(Fchoose);           % 归一化到截距

% 4) 先把已选整层关联到参考向量，得到壁龛占用 rho
SelMask  = false(size(ChooseIdx));
SelMask(1:numel(idxNext)) = true;               % 前 numel(idxNext) 行是已选整层
FN_sel   = FN(SelMask,:);                       % 可能为空
if isempty(FN_sel)
    rho = zeros(size(V,1),1);
else
    [sel_niche, ~] = associate_to_ref(FN_sel, V);
    if isempty(sel_niche)
        rho = zeros(size(V,1),1);
    else
        rho = accumarray(sel_niche, 1, [size(V,1) 1]);
    end
end

% 5) 末层关联 + 按配额填满
FN_last  = FN(~SelMask,:);                      % 对应 Last 的那部分
[last_niche, last_dperp] = associate_to_ref(FN_last, V);

buckets = cell(size(V,1),1);
for i = 1:numel(idxLast)                        % 注意用 idxLast 的长度
    buckets{last_niche(i)} = [buckets{last_niche(i)}; i];
end

ChosenFromLast = false(numel(idxLast),1);
while sum(ChosenFromLast) < Kneed
    minrho = min(rho);
    J = find(rho == minrho);
    picked = false;
    for jj = 1:numel(J)
        j = J(jj);
        cand = buckets{j};
        cand = cand(~ChosenFromLast(cand));
        if isempty(cand), continue; end
        if rho(j) == 0
            [~,ix] = min(last_dperp(cand));
            idxPick = cand(ix);
        else
            idxPick = cand(randi(numel(cand)));
        end
        ChosenFromLast(idxPick) = true;
        rho(j) = rho(j) + 1;
        picked = true;
        if sum(ChosenFromLast) >= Kneed, break; end
    end
    if ~picked
        rest = find(~ChosenFromLast);
        if isempty(rest), break; end
        take = min(Kneed - sum(ChosenFromLast), numel(rest));
        ridx = rest(randperm(numel(rest), take));
        ChosenFromLast(ridx) = true;
    end
end

Select = Next(:);                                 % 1) 列向量化
tmp    = false(size(PopObj,1),1);
tmp(idxLast(ChosenFromLast)) = true;
Select = Select | tmp;                            % 2) 列向量 OR
Select = Select(:);                               % 3) 再保险，保持 N×1
Population = PopBoth(Select);
end


%% ---------- NSGA-III 归一化：理想点 + 极点/截距兜底 ----------
function [zmin, FN] = normalize_nsga3(F)
% F: K×M  原始目标
zmin = min(F, [], 1);
Fp = bsxfun(@minus, F, zmin);              % 平移到理想点
M  = size(F,2);

% 找极点（ASF）
W = eye(M)*1e-6 + diag(ones(M,1)-1e-6);   % 每次强调一个目标
asf = zeros(size(F,1), M);
for i=1:M
    wi = W(i,:);
    asf(:,i) = max(bsxfun(@rdivide, Fp, wi), [], 2);
end
[~,extIdx] = min(asf, [], 1);             % 每个目标的极点索引
E = Fp(extIdx, :);                         % M×M

% 计算截距 a
useMax = false;
if rank(E) == M
    alpha = E \ ones(M,1);
    a = 1 ./ alpha';
    if any(~isfinite(a)) || any(a<=0)
        useMax = true;
    end
else
    useMax = true;
end
if useMax
    a = max(Fp, [], 1);
end
a(a==0) = 1;                               % 防零

FN = bsxfun(@rdivide, Fp, a);              % 归一化
end

%% ---------- 关联到参考向量：返回壁龛编号与垂直距离 ----------
function [niche, dperp] = associate_to_ref(FN, V)
% FN: K×M（归一化目标）；V: R×M（单位参考向量）
if isempty(FN)
    niche = zeros(0,1); dperp = zeros(0,1); return;
end
normF = sqrt(sum(FN.^2, 2)); normF(normF==0) = eps;
Cos = (FN * V') ./ normF;                   % cos(theta)，因 |V|=1
Cos = max(min(Cos,1),-1);
dperp_all = normF .* sqrt(1 - Cos.^2);
[dperp, niche] = min(dperp_all, [], 2);
end
