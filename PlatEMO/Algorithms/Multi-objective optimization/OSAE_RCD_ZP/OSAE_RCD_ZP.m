classdef OSAE_RCD_ZP < ALGORITHM
% <2025> <multi/many> <real/integer/label/binary/permutation> <large>
% OSAE_RCD_ZP : Zero-Param Row–Column Dual-Drive with change (O–S–D)
% - 行→定列：目标列间图谱分解，eigengap 自动给 K* 与 O_k
% - 列→定行：由 T 子块奇异谱给注入规模与方向（零/弱奇异子空间）
% - change：Gamma_C>0 选列相位；否则 Q>0 选行相位；否则列相位
% - C-phase：组内 PCA 取 r=rank_tau，潜空间“中点”→ 分组写回（闭式步）
% - D-phase：NSE(ker(T)) 最大可行步 + ACP(球面测地中点)，仅改选中行
%
% 示例：
%   main('-algorithm',@OSAE_RCD_ZP,'-problem',@DTLZ3,'-M',10,'-D',200, ...
%        '-N',210,'-evaluation',1e5);
%
% 依赖：PlatEMO 的 NDSort / INDIVIDUAL / Problem.Evaluation 等标准接口。
%
% 说明：本实现遵循“零新增超参”原则，除机器精度阈值外不引入额外参数。

    methods
        function main(obj, Problem)
            % 初始化
            Population = Problem.Initialization();
            N = Problem.N; D = Problem.D; M = size(Population.objs,2);
            lastPhase = 'C';   % 用于平局时交替
            tau = eps*max([N,M,D]);  % SVD 截断阈值（LAPACK 风格）

            while obj.NotTerminated(Population)
                % 读出当前矩阵
                X = Population.decs;   % N×D
                Y = Population.objs;   % N×M

                % —— 拟合 T（zscore 截断伪逆），行→定列：分组 O_k
                map = fitMap_pinv_tau(Y, X, tau);
                O_groups = spectral_group_from_Y(Y);    % 自动 K* 与 O_k
                S_groups = hard_assign_S(map.T, O_groups, D);  % 硬唯一指派

                % —— 列→定行：统计自由度 Q 与各组零/弱奇异子空间基
                [Q, NullBases] = nullspaces_from_T(map.T, O_groups, S_groups, tau);

                % —— 评估列相位的“可下降度” Gamma_C（潜空间中点）
                Gamma_C = compute_gammaC(Y, X, map, O_groups, S_groups);

                % ========== change：无参裁决 ==========
                if Gamma_C > 0
                    phase = 'C';
                elseif Q > 0
                    phase = 'D';
                else
                    phase = 'C';
                end
                if Gamma_C <= 10*eps && Q > 0 && lastPhase=='C'
                    % 平局时与上代相反，轻度交替
                    phase = 'D';
                end

                % ========== 执行该相位 ==========
                switch phase
                    case 'C'   % 列相位：组内 PCA 中点 → 分组写回（闭式步）
                        OffDec = X;
                        for k = 1:numel(O_groups)
                            Ok = O_groups{k}; Sk = S_groups{k};
                            if isempty(Ok) || isempty(Sk), continue; end
                            Yk = Y(:, Ok);
                            Yk_new = pca_midpoint(Yk, tau);  % 潜空间中点（固定 0.5）
                            Xhat_Sk = apply_map_sub(map, Yk_new, Ok, Sk);
                            Delta = Xhat_Sk - OffDec(:, Sk);
                            % 闭式步（此处等价于 eta=1）
                            OffDec(:, Sk) = OffDec(:, Sk) + Delta;
                        end
                        OffDec = min(max(OffDec, Problem.lower), Problem.upper);
                        Offspring = Problem.Evaluation(OffDec);

                    case 'D'   % 行相位：NSE 最大可行步 + ACP 测地中点
                        % 注入规模 nInject = Q；从角最拥挤样本里选
                        nInject = max(1, min(N, Q));
                        idxInject = pick_crowded_rows(Y, nInject);

                        OffDec = X;   % 从当前解出发
                        % —— NSE：沿各组 NullBasis 在 Sk 列做最大可行步
                        for t = 1:numel(idxInject)
                            i = idxInject(t);
                            for k = 1:numel(O_groups)
                                Sk = S_groups{k};
                                if isempty(Sk), continue; end
                                Vk = NullBases{k};         % |Sk|×q
                                if isempty(Vk), continue; end
                                u = Vk(:,1)';              % 固定选第一基向量（零参）
                                OffDec(i, Sk) = max_feasible_step(OffDec(i, Sk), u, ...
                                    Problem.lower(Sk), Problem.upper(Sk));
                            end
                        end

                        % —— ACP：球面测地中点（anchors：极点 + 最远点补充）
                        anchors = build_anchors(Y);
                        for t = 1:numel(idxInject)
                            i = idxInject(t);
                            a = pick_farthest_anchor(Y(i,:), anchors);
                            ymid = geodesic_midpoint(Y(i,:), a);
                            % 按组写回（闭式步）
                            for k = 1:numel(O_groups)
                                Ok = O_groups{k}; Sk = S_groups{k};
                                if isempty(Ok) || isempty(Sk), continue; end
                                yk = ymid(:, Ok);
                                xhat = apply_map_sub(map, yk, Ok, Sk);
                                OffDec(i, Sk) = xhat(1,:); % 这里等价于 eta=1
                            end
                        end
                        OffDec = min(max(OffDec, Problem.lower), Problem.upper);
                        Offspring = Problem.Evaluation(OffDec);
                end

                % 选择：非支配 + 角度截断
                Population = env_select_angle([Population, Offspring], Problem.N);
                lastPhase = phase;
            end
        end
    end
end

%% ===================== 线性映射与分组 =====================
function map = fitMap_pinv_tau(Y, X, tau)
% 截断伪逆：T = pinv_tau(Yz)*Xz；并保存标准化统计
muY = mean(Y,1);  sigY = std(Y,0,1);  sigY(sigY==0) = 1;
muX = mean(X,1);  sigX = std(X,0,1);  sigX(sigX==0) = 1;
Yz = (Y - muY)./sigY;
Xz = (X - muX)./sigX;
[U,S,V] = svd(Yz,'econ'); s = diag(S);
sInv = zeros(size(s)); sInv(s >= tau) = 1./s(s >= tau);
G = V * diag(sInv) * U';     % pinv_tau(Yz)
T = G * Xz;                  % M×D
map.T = T; map.muY = muY; map.sigY = sigY; map.muX = muX; map.sigX = sigX;
end

function O_groups = spectral_group_from_Y(Y)
% 目标列之间的谱聚类：K* 用 eigengap
[N,M] = size(Y);
if M==1, O_groups = {1}; return; end
% 列归一化→余弦相似度（非负）
Yz = zscore(Y,0,1); Yz(~isfinite(Yz)) = 0;
U = Yz ./ max(sqrt(sum(Yz.^2,1)), eps);    % N×M 列单位化
W = max(0, U.'*U);                         % M×M 非负余弦
d = sum(W,2); Dm = diag(d + eps);
Lsym = eye(M) - (Dm^(-1/2))*W*(Dm^(-1/2));
Lsym = (Lsym+Lsym')/2;
[V,E] = eig(Lsym); [lam, ord] = sort(diag(E),'ascend'); V = V(:,ord);
% eigengap：1..M-1
gaps = lam(2:end) - lam(1:end-1);
[~,kstar] = max(gaps); K = max(1,min(M-1,kstar));
H = V(:,1:K);              % M×K
% kmeans 划分目标列（需要统计工具箱）
lbl = kmeans(H, K, 'Replicates', 5, 'MaxIter', 200, 'Display', 'off');
O_groups = cell(1,K);
for k = 1:K, O_groups{k} = find(lbl==k).'; end
end

function S_groups = hard_assign_S(T, O_groups, D)
% 硬唯一指派：每个决策列归属能量最大的组
K = numel(O_groups);
energy = zeros(K, D);
for k = 1:K
    Ok = O_groups{k};
    if isempty(Ok), continue; end
    Tk = T(Ok, :);                  % |Ok|×D
    energy(k, :) = sqrt(sum(Tk.^2, 1));
end
[~, owner] = max(energy, [], 1);
S_groups = cell(1, K);
for k = 1:K
    S_groups{k} = find(owner==k).';
end
end

function [Q, NullBases] = nullspaces_from_T(T, O_groups, S_groups, tau)
% 统计自由度 Q 与各组 Sk 列上的零/弱奇异子空间基（右奇异向量）
K = numel(O_groups); NullBases = cell(1,K); Q = 0;
for k = 1:K
    Ok = O_groups{k}; Sk = S_groups{k};
    if isempty(Ok) || isempty(Sk), NullBases{k} = []; continue; end
    Tk = T(Ok, Sk);
    if isempty(Tk), NullBases{k} = []; continue; end
    [~,S,V] = svd(Tk,'econ'); s = diag(S);
    keep = find(s <= tau);
    if isempty(keep), NullBases{k} = []; continue; end
    NullBases{k} = V(:, keep);     % |Sk|×q
    Q = Q + numel(keep);
end
end

%% ===================== 列相位：潜空间中点 & 写回 =====================
function Yk_new = pca_midpoint(Yk, tau)
% 组内 PCA：取 r=rank_tau，在潜空间与中心点做 0.5 中点
[Yz, mu, sg] = safe_zscore(Yk);
[U,S,V] = svd(Yz,'econ'); s = diag(S);
r = nnz(s >= tau); r = max(1, min(r, size(V,2)));
W = V(:,1:r);
Z = Yz * W;                % N×r
zc = mean(Z,1);
Znew = 0.5*(Z + zc);       % 中点（固定 0.5）
Yk_new = (Znew * W').*sg + mu;
end

function Xhat_Sk = apply_map_sub(map, Yk_new, Ok, Sk)
% 仅写回 Sk 列
Yz = bsxfun(@rdivide, bsxfun(@minus, Yk_new, map.muY(Ok)), map.sigY(Ok));
Xz = Yz * map.T(Ok, Sk);
Xhat_Sk = bsxfun(@plus, bsxfun(@times, Xz, map.sigX(Sk)), map.muX(Sk));
end

function Gamma_C = compute_gammaC(Y, X, map, O_groups, S_groups)
% 以潜空间中点产生的 Delta 估算列相位“可下降度” Γ_C
Gamma_C = 0;
for k = 1:numel(O_groups)
    Ok = O_groups{k}; Sk = S_groups{k};
    if isempty(Ok) || isempty(Sk), continue; end
    Yk = Y(:, Ok); Yk_new = pca_midpoint(Yk, eps*max(size(Y)));
    Xhat_Sk = apply_map_sub(map, Yk_new, Ok, Sk);
    Delta = Xhat_Sk - X(:, Sk);
    Gamma_C = Gamma_C + sum(Delta(:).*Delta(:)); % 等价于 ||Delta||_F^2
end
end

%% ===================== 行相位：NSE & ACP =====================
function idx = pick_crowded_rows(Y, n)
% 角最拥挤：每行与最近邻的角距离最小者优先
Yh = normalize_rows(Y); N = size(Yh,1);
Cos = Yh*Yh.'; Cos(1:N+1:end) = -inf;        % 自身置 -inf
minAngle = acos(max(min(max(Cos,[],2),1),-1)); % 最近邻角
[~, ord] = sort(minAngle, 'ascend');          % 角距越小越拥挤
n = min(n, N); idx = ord(1:n);
end

function anchors = build_anchors(Y)
% 锚点：每目标的极点 + farthest-first 扩充到 min(2M, N)
[N,M] = size(Y);
Yh = normalize_rows(Y);
% 极点（最小化假设）：每列最小行
sel = zeros(1,M);
for m = 1:M
    [~, sel(m)] = min(Y(:,m));
end
sel = unique(sel, 'stable');
% farthest-first 扩充
targetK = min(2*M, N);
while numel(sel) < targetK
    % 选与当前集合的最大最小角者
    S = Yh(sel,:); rest = setdiff(1:N, sel);
    best = -inf; besti = rest(1);
    for r = rest
        cmin = max(S*Yh(r,:)'); % cos 最大即角最小
        score = -cmin;          % 想要角尽量大 => cos 尽量小
        if score > best
            best = score; besti = r;
        end
    end
    sel(end+1) = besti; %#ok<AGROW>
end
anchors = Y(sel, :);
end

function a = pick_farthest_anchor(y, anchors)
yh = normalize_rows(y); Ah = normalize_rows(anchors);
cosv = Ah * yh'; [~, ix] = min(cosv); a = anchors(ix, :);
end

function ymid = geodesic_midpoint(y, a)
% 球面测地中点，再缩放回 |y| 的半径
yh = normalize_rows(y); ah = normalize_rows(a);
mid = yh + ah; mid = mid ./ max(norm(mid), eps);
ymid = mid * norm(y);   % 保留原半径
end

function xr = max_feasible_step(xrow, u, lower, upper)
% 沿方向 u 走最大可行正步（命中边界即停）
% 允许 u 的正负；取能保持所有分量在 [lower, upper] 内的最大 lambda
lambda = inf;
for j = 1:numel(xrow)
    if u(j) > 0
        lambda = min(lambda, (upper(j) - xrow(j))/u(j));
    elseif u(j) < 0
        lambda = min(lambda, (lower(j) - xrow(j))/u(j));
    end
end
if ~isfinite(lambda), lambda = 0; end
xr = xrow + lambda * u;
end

%% ===================== 选择：非支配 + 角度截断 =====================
function Population = env_select_angle(PopBoth, N)
PopObj = PopBoth.objs; [FrontNo, MaxFNo] = NDSort(PopObj, N);
Next = FrontNo < MaxFNo;
Last = find(FrontNo == MaxFNo); K = N - sum(Next);
if K > 0
    Choose = truncation_angle(PopObj(Last, :), K);
    Next(Last(Choose)) = true;
end
Population = PopBoth(Next);
end

function Choose = truncation_angle(PopObj, K)
% 目标归一化→单位化→按角度尽量分散地选 K 个
fmax = max(PopObj, [], 1); fmin = min(PopObj, [], 1);
span = fmax - fmin; span(span==0) = 1;
P = (PopObj - fmin) ./ span;
nrm = sqrt(sum(P.^2, 2)); nrm(nrm==0) = 1; U = P ./ nrm;
Cos = U*U.'; Cos(1:size(Cos,1)+1:end) = -inf;
Choose = false(1, size(P,1));
% 先选每个目标的极点（极大分散）
[~, extreme] = max(P, [], 1); Choose(extreme) = true;
while sum(Choose) < K
    unSel = find(~Choose);
    sel   = find(Choose);
    if isempty(sel)
        [~,x] = min(max(Cos,[],2));
        Choose(x) = true;
        continue;
    end
    % 选使得与已选集合的最大 cos 最小的点
    best = -inf; besti = unSel(1);
    for t = 1:numel(unSel)
        r = unSel(t);
        cmax = max(Cos(r, sel));
        score = -cmax;
        if score > best
            best = score; besti = r;
        end
    end
    Choose(besti) = true;
end
end

%% ===================== 小工具 =====================
function [Z, mu, sg] = safe_zscore(X)
mu = mean(X,1); sg = std(X,0,1); sg(sg==0) = 1; Z = (X - mu) ./ sg;
end

function Yh = normalize_rows(Y)
% 行单位化（零向量保持零）
nrm = sqrt(sum(Y.^2, 2));
nrm(nrm==0) = 1;
Yh = Y ./ nrm;
end
