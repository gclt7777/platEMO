classdef BAXOSD_K3 < ALGORITHM
% <2025> <multi/many> <real/integer/label/binary/permutation> <large>
% BAXOSD_K3 : Bi-Axis eXploration with O–S–D
%
% 核心不变：
%   - C-phase（列分组/收敛）：目标列分组 O_k → 组内线性 AE(PCA) 的潜空间 latent-DE → 仅写回对应 S_k（半步融合）
%   - D-phase（行分组/多样）：参考向量扇区（按行）→ 扇区内目标空间 DE → Δy·T 全列写回（半步融合）
%   - 环境选择：RVEA-APD（可行性优先），theta = (FE/maxFE)^alpha
%
% 仅暴露参数：
%   alpha --- 1.6 --- 进度调度陡峭度（唯一外参）

    methods
        function main(Algorithm,Problem)
            %% 参数设置（公开 alpha + 参考向量频率 fr）
            [alpha, fr] = Algorithm.ParameterSet(1.6, inf);

            % 决策列中 A 类变量的比例
            rhoA          = 0.5;   % A 占总决策列的比例
            periodGroup   = 10;    % 每 10 代重分组/重指派
            EVR_target    = 0.85;  % 组内 PCA 解释率阈
            rk_cap_minmax = [1,4]; % PCA 维度夹取范围
            kappa_tar     = 1e6;   % T 条件数限幅目标
            eta           = 0.5;   % 半步融合

            %% 参考向量 & 初始种群（路线 A：严格尊重 Problem.N）
            N0  = Problem.N;                       % 记录用户设定的种群规模
            [V0,~] = UniformPoint(N0,Problem.M);  % 只取参考向量，不改写 Problem.N
            V   = V0;

            Population = Problem.Initialization();  % 这里的规模 = N0

            % 预计算 gamma（参考向量最小夹角）：按 RVEA 每代计算亦可，这里缓存一次
            cosineVV = 1 - pdist2(V,V,'cosine');
            cosineVV(1:size(cosineVV,1)+1:end) = 0;
            gammaV = min(acos(max(-1,min(1,cosineVV))),[],2); 

            % 初始线性映射 T（z-score 域）
            map = fit_T_linear_zscore(Population.objs, Population.decs, kappa_tar);

            % 初始 O-S-D 分组（g = 1）
            [O_groups,K] = group_objectives_angle(Population.objs, Problem.M); %#ok<ASGLU>
            S_groups     = design_S_AP_balanced(map.T, O_groups, rhoA);
            curGen       = 1;

            %% 主循环
            while Algorithm.NotTerminated(Population)
                theta  = (Problem.FE/Problem.maxFE)^alpha;
                curGen = max(1,ceil(Problem.FE/Problem.N));

                % ------- 参考向量自适应（简单按频率） -------
                if ~isinf(fr)
                    freq = max(1,ceil(fr*Problem.maxFE/Problem.N));
                    if ~mod(curGen, freq)
                        % 注意：这里不再假设 size(V,1) >= Problem.N，直接整体更新
                        V = ReferenceVectorAdaptation(Population.objs,V0);
                        cosineVV = 1 - pdist2(V,V,'cosine');
                        cosineVV(1:size(cosineVV,1)+1:end) = 0;
                        gammaV   = min(acos(max(-1,min(1,cosineVV))),[],2); %#ok<NASGU>
                    end
                end

                % ------- 每代拟合线性映射 T（z-score 域） -------
                map = fit_T_linear_zscore(Population.objs, Population.decs, kappa_tar);

                % ------- 周期性重分组与唯一指派（从第 2 代开始） -------
                if curGen > 1 && mod(curGen-1,periodGroup)==0
                    [O_groups,K] = group_objectives_angle(Population.objs, Problem.M); %#ok<ASGLU>
                    S_groups     = design_S_AP_balanced(map.T, O_groups, rhoA);
                end

                % ------- 相位配比（随 theta 派生） -------
                rC = 0.40 + 0.40*theta;
                nC = round(Problem.N * rC);
                nD = Problem.N - nC;

                % ---------- C-phase: 组内 PCA + latent-DE，仅写回 S_k ----------
                OffC = Operator_CPhase(Population, nC, O_groups, S_groups, map, ...
                                       EVR_target, rk_cap_minmax, theta, eta, Problem);

                % ---------- D-phase: 扇区内目标空间 DE，Δy·T 全列写回 ----------
                OffD = Operator_DPhase(Population, nD, V, gammaV, map, theta, eta, Problem);

                % ---------- RVEA-APD 环境选择 ----------
                nBase = length(Population);
                nCnow = length(OffC);
                nDnow = length(OffD);
                Population = EnvironmentalSelection_BAXOSD([Population,OffC,OffD], V, theta, ...
                                          nBase, nCnow, nDnow, Problem.N);
            end
        end
    end
end

%% ======================= S 设计：score + 排序 + 均衡 =======================

function S_groups = design_S_AP_balanced(T, O_groups, rhoA)
% 基于 T 子块能量的 A/P + S_k 设计（"大刀阔斧版"）
%
% 输入：
%   - T        : M x D 的线性逆映射矩阵
%   - O_groups: 1 x K 的 cell，每个元素是该组内目标列索引
%   - rhoA     : A 类变量占比（0~1）
%
% 输出：
%   - S_groups : 1 x K 的 cell，每个元素是该组内决策列索引
%
% 说明：这里已经去掉分组监控用的 stats，仅保留分组本身。

    K = numel(O_groups);
    if K == 0
        S_groups = {};
        return;
    end

    [~,D] = size(T);
    if D == 0
        S_groups = cell(1,K);
        return;
    end

    % -------- 1) 计算每个决策列 j 在各组上的能量 e_{k,j} --------
    energy = zeros(K,D);
    for k = 1:K
        Ok = O_groups{k};
        if isempty(Ok)
            continue;
        end
        Tk = T(Ok,:);             % |Ok| x D
        energy(k,:) = sum(Tk.^2,1);
    end

    % 总能量 & 最大能量（用于 score）
    E_total = sum(energy,1);
    eps_val = 1e-12;
    E_total(E_total < eps_val) = eps_val;

    [e1, ~] = max(energy,[],1);   % 最大能量
    score   = e1 ./ E_total;      % 集中度：越接近 1 越"铁 A"

    % -------- 2) 按 score 排序，固定比例划分 A/P --------
    if D <= 1
        KA = 1;       % 极小维度兜底：全 A
    else
        KA = round(rhoA * D);
        KA = max(1, min(D-1, KA));  % 保证 1 <= A <= D-1
    end

    [~, order] = sort(score,'descend');
    A_mask = false(1,D);
    A_mask(order(1:KA)) = true;
    P_mask = ~A_mask;

    A_idx = find(A_mask);
    P_idx = find(P_mask);

    % -------- 3) 为每个列 j 找到 g1/g2（前两个能量最大的组） --------
    g1 = ones(1,D);
    g2 = ones(1,D);

    if K == 1
        g1(:) = 1; g2(:) = 1;
    else
        % 对每列按能量降序排序，取前两名
        [~, idxSortAll] = sort(energy, 1, 'descend');  % K x D
        g1 = idxSortAll(1,:);
        if K >= 2
            g2 = idxSortAll(2,:);
        else
            g2 = g1;
        end
    end

    % -------- 4) A 阶段：先立骨架，控制每组 A 的配额 --------
    S_groups = cell(1,K);
    lenS     = zeros(1,K);   % 当前 S_k 的列数
    countA   = zeros(1,K);   % 当前每组 A 的数量

    if ~isempty(A_idx)
        % A 内部仍按 score 从大到小处理（优先分配"最 A 的列"）
        [~, ordA] = sort(score(A_idx),'descend');
        A_idx_sorted = A_idx(ordA);

        targetA = ceil(numel(A_idx) / K);  % 每组 A 的目标配额（均匀分配）

        for jj = 1:numel(A_idx_sorted)
            j = A_idx_sorted(jj);
            c1 = g1(j);
            c2 = g2(j);

            if K == 1 || c1 == c2
                kbest = c1;
            else
                % 优先让没到 targetA 的组多吃 A
                if countA(c1) >= targetA && countA(c2) < targetA
                    kbest = c2;
                elseif countA(c2) >= targetA && countA(c1) < targetA
                    kbest = c1;
                else
                    % 否则在两候选中选 A 数更少的一个
                    if countA(c1) <= countA(c2)
                        kbest = c1;
                    else
                        kbest = c2;
                    end
                end
            end

            S_groups{kbest} = [S_groups{kbest}, j];
            lenS(kbest)     = lenS(kbest) + 1;
            countA(kbest)   = countA(kbest) + 1;
        end
    end

    % -------- 5) P 阶段：始终把列塞给"当前最瘦的组" --------
    if ~isempty(P_idx)
        % 对 P 也可以按 score 排序（高 score 的优先被放置到"最紧缺的组"）
        [~, ordP] = sort(score(P_idx),'descend');
        P_idx_sorted = P_idx(ordP);

        for jj = 1:numel(P_idx_sorted)
            j = P_idx_sorted(jj);

            % 找当前最瘦的组
            minLen = min(lenS);
            candGroups = find(lenS == minLen);

            % 在这些最瘦的组中，选 energy 最大的组
            if numel(candGroups) == 1
                kbest = candGroups;
            else
                e_cand = energy(candGroups,j);
                [~, posBest] = max(e_cand);
                kbest = candGroups(posBest);
            end

            S_groups{kbest} = [S_groups{kbest}, j];
            lenS(kbest)     = lenS(kbest) + 1;
        end
    end
end

%% ======================= 环境选择 =======================
function Population = EnvironmentalSelection_BAXOSD(PopAll,V,theta,nBase,nC,nD,Ntar,varargin)
% RVEA-APD 环境选择（可行优先，每参考向量先留 1）
% 若选不满 Ntar，再从剩余个体中按 APD 补满，优先补 C/D 个体
%
% 说明：早期版本存在用于控制调试输出的 verboseLog 标志。为兼容遗留调用，
% 此处接受多余的可变参数但完全忽略（环境选择阶段现已无任何日志逻辑）。

    % 明确占位以兼容仍携带 verboseLog 的调用，避免未定义变量错误
    verboseLog = false; %#ok<NASGU>
    if ~isempty(varargin)
        extra = varargin{1};
        if islogical(extra) && isscalar(extra)
            verboseLog = extra; %#ok<NASGU>
        end
    end

    PopObj = PopAll.objs;
    [Ncand,M]  = size(PopObj);
    NV     = size(V,1);
    Ntar   = min(Ntar,Ncand);   % 不要超过候选总数

    % 平移到原点（RVEA 口径）
    PopObjShift = PopObj - repmat(min(PopObj,[],1),Ncand,1);

    % 约束违反度（可行优先）
    if isempty(PopAll.cons)
        CV = zeros(Ncand,1);
    else
        CV = sum(max(0,PopAll.cons),2);
    end

    % 参考向量最小夹角 gamma
    cosineVV = 1 - pdist2(V,V,'cosine');
    cosineVV(1:size(cosineVV,1)+1:end) = 0;
    gamma = min(acos(max(-1,min(1,cosineVV))),[],2);

    % 关联到参考向量
    Angle = acos(max(0,1 - pdist2(PopObjShift,V,'cosine'))); % 数值安全
    [~,associate] = min(Angle,[],2);

    % ---------- 第一步：每扇区先挑 1 个 ----------
    pick = zeros(1,NV);
    for i = unique(associate)'
        idxFea = find(associate==i & CV==0);
        idxInf = find(associate==i & CV~=0);
        if ~isempty(idxFea)
            APD = (1+M*theta*Angle(idxFea,i)/gamma(i)) .* ...
                  sqrt(sum(PopObjShift(idxFea,:).^2,2));
            [~,best] = min(APD);
            pick(i)  = idxFea(best);
        elseif ~isempty(idxInf)
            [~,best] = min(CV(idxInf));
            pick(i)  = idxInf(best);
        end
    end

    selIdx = unique(pick(pick~=0));   % 先拿到一扇区一名额的集合

    % ---------- 第二步：不足 Ntar 时，按 APD 补满，优先 C/D ----------
    if numel(selIdx) > Ntar
        % 选多了：用 APD 再截断一次
        keepIdx  = selIdx;
        score    = local_apd_score(keepIdx,PopObjShift,Angle,associate,gamma,CV,M,theta);
        [~,ord]  = sort(score);
        selIdx   = keepIdx(ord(1:Ntar));
    elseif numel(selIdx) < Ntar
        need      = Ntar - numel(selIdx);
        allIdx    = 1:Ncand;
        restIdx   = setdiff(allIdx,selIdx);

        scoreRest = local_apd_score(restIdx,PopObjShift,Angle,associate,gamma,CV,M,theta);

        % 分成 base / C / D 三组
        isBase = restIdx <= nBase;
        isC    = restIdx > nBase        & restIdx <= nBase+nC;
        isD    = restIdx > nBase+nC;    % 剩下的全归 D

        idxB = restIdx(isBase); scB = scoreRest(isBase);
        idxC = restIdx(isC);    scC = scoreRest(isC);
        idxD = restIdx(isD);    scD = scoreRest(isD);

        % 各组内部按 APD 从小到大排
        [~,oC] = sort(scC);  idxC = idxC(oC);
        [~,oD] = sort(scD);  idxD = idxD(oD);
        [~,oB] = sort(scB);  idxB = idxB(oB);

        % 优先补 C，其次 D，最后 base
        order = [idxC, idxD, idxB];
        extra = order(1:min(need,numel(order)));

        selIdx = [selIdx, extra];
    end

    if verboseLog
        sb = sum(selIdx <= nBase);
        sc = sum(selIdx >  nBase        & selIdx <= nBase+nC);
        sd = sum(selIdx >  nBase+nC);

        feasSel = sum(CV(selIdx) == 0);
        feasAll = sum(CV == 0);

        fprintf('[BAX_ENV][g%04d] t=%4.3f N=%d/%d | c=%d/%d/%d | s=%d/%d/%d | u=%d/%d\n', ...
                curGen, theta, numel(selIdx), Ntar, nBase, nC, nD, sb, sc, sd, feasSel, feasAll);
    end

    Population = PopAll(selIdx);
end


% ====== 计算一批个体的 APD 分数（方便复用） ======
function score = local_apd_score(idx,PopObjShift,Angle,associate,gamma,CV,M,theta)
    if isempty(idx)
        score = [];
        return;
    end
    k = numel(idx);
    score = zeros(k,1);
    for t = 1:k
        j  = idx(t);
        if CV(j) == 0
            i  = associate(j);
            apd = (1+M*theta*Angle(j,i)/gamma(i)) * ...
                  sqrt(sum(PopObjShift(j,:).^2));
            score(t) = apd;
        else
            % 不可行解给一个大罚值
            score(t) = 1e6 + CV(j);
        end
    end
end


%% ======================= 参考向量缩放 =======================

function V = ReferenceVectorAdaptation(PopObj,V0)
% 参考向量线性缩放（RVEA 简版）
    span = max(PopObj,[],1)-min(PopObj,[],1);
    V    = V0 .* repmat(span,size(V0,1),1);
end

%% ======================= C-phase =======================

function Off = Operator_CPhase(Pop, nC, O_groups, S_groups, map, EVR_target, rk_cap_minmax, theta, eta, Problem)
% 列相位：组内 PCA + latent-DE，仅在对应 S_k 上写回

    if nC<=0
        Off = Pop.empty(); return;
    end
    X = Pop.decs; Y = Pop.objs;
    [N,~] = size(Y);
    D = size(X,2); %#ok<NASU>

    % 只选同时满足 O_k、S_k 均非空的组
    valid = find(~cellfun(@isempty,O_groups) & ~cellfun(@isempty,S_groups));
    if isempty(valid)
        Off = Pop.empty(); return;
    end

    % 随 theta 调度 latent DE 参数
    F_l  = max(0.40, min(0.75, 0.75 - 0.35*theta));
    CR_l = max(0.50, min(0.95, 0.95 - 0.45*theta));

    OffX = zeros(nC,size(X,2));

    for t = 1:nC
        k  = valid(randi(numel(valid)));
        Oi = O_groups{k};
        Si = S_groups{k};
        i  = randi(N);

        % PCA 基
        [Wk, mu_k, rk] = pca_basis(Y(:,Oi), EVR_target, rk_cap_minmax);
        if rk == 0 || isempty(Wk)
            OffX(t,:) = X(i,:);
            continue;
        end

        % latent-DE in PCA space
        r1 = randi(N); r2 = randi(N); r3 = randi(N);
        zi = Wk'*(Y(i,Oi) - mu_k)';
        z1 = Wk'*(Y(r1,Oi) - mu_k)';
        z2 = Wk'*(Y(r2,Oi) - mu_k)';
        z3 = Wk'*(Y(r3,Oi) - mu_k)';

        v  = z1 + F_l*(z2 - z3);
        u  = zi;
        jrand = randi(rk);
        for j = 1:rk
            if rand < CR_l || j==jrand, u(j) = v(j); end
        end

        y_old_k = Y(i,Oi);
        y_new_k = (mu_k + (Wk*u)');

        % Δy → Δx（只写回 S_k 列）
        dyz = (y_new_k - y_old_k) ./ map.sigY(Oi);
        dXZ = dyz * map.T(Oi,Si);
        dX  = dXZ .* map.sigX(Si);

        xCand      = X(i,:);
        xCand(Si)  = X(i,Si) + eta * dX;
        xCand      = min(max(xCand,Problem.lower),Problem.upper);
        OffX(t,:)  = xCand;
    end

    Off = Problem.Evaluation(OffX);
end

%% ======================= D-phase =======================

function Off = Operator_DPhase(Pop, nD, V, gammaV, map, theta, eta, Problem) %#ok<INUSD>
% 行相位：扇区内目标空间 DE → Δy·T 全列写回（半步融合）

    if nD<=0
        Off = Pop.empty();
        return;
    end
    X = Pop.decs; Y = Pop.objs;
    [N,M] = size(Y);
    D = size(X,2);

    % 约束信息：优先在可行扇区采样
    if isempty(Pop.cons)
        CV = zeros(N,1);
    else
        CV = sum(max(0,Pop.cons),2);
    end
    feasibleMask = CV==0;

    % 扇区归属（按平移后角度）
    Y0 = Y - repmat(min(Y,[],1),N,1);
    Angle = acos(max(0,1 - pdist2(Y0,V,'cosine')));
    [~,sector] = min(Angle,[],2);

    Fd  = 0.90 - 0.40*theta; Fd = max(0.50,min(0.90,Fd));
    CRd = 0.60 + 0.30*theta; CRd = max(0.60,min(0.90,CRd));

    OffX = zeros(nD,D);

    for t = 1:nD
        % 1) 选一个扇区（按人口加权）
        sList = unique(sector)';
        countsAll = arrayfun(@(s) sum(sector==s), sList);
        countsFea = arrayfun(@(s) sum(sector==s & feasibleMask), sList);
        if any(countsFea>0)
            weights = countsFea;
        else
            weights = countsAll;
        end
        s = randsample(sList,1,true,weights);

        % 扇区内样本索引
        idx = find(sector==s);
        if any(feasibleMask(idx))
            idx = idx(feasibleMask(idx));
        end
        if numel(idx) < 3
            idx = find(feasibleMask);
        end
        if numel(idx) < 3
            idx = 1:N; % 兜底：全局抽
        end

        % 2) 目标个体 i 与父代 r1,r2,r3
        i  = idx(randi(numel(idx)));
        r1 = idx(randi(numel(idx)));
        r2 = idx(randi(numel(idx)));
        r3 = idx(randi(numel(idx)));

        % 3) 目标空间 DE/rand/1/bin
        v  = Y(r1,:) + Fd*(Y(r2,:) - Y(r3,:));
        u  = Y(i,:);
        jrand = randi(M);
        for j = 1:M
            if rand < CRd || j==jrand
                u(j) = v(j);
            end
        end
        y_off = u;

        % 4) 写回：Δy → z-score → Δx = Δy_z * T → 原域半步融合
        dy  = y_off - Y(i,:);
        dyz = dy ./ map.sigY(:)';            % z-score 差分
        dXZ = dyz * map.T;                   % 全矩阵写回（z 域）
        dX  = dXZ .* map.sigX(:)';           % 原域增量
        xCand = X(i,:) + eta * dX;

        % 5) 边界修复
        xCand = min(max(xCand,Problem.lower),Problem.upper);
        OffX(t,:) = xCand;
    end

    Off = Problem.Evaluation(OffX);
end

%% ======================= PCA basis =======================

function [W,mu,rc] = pca_basis(Y, EVR_target, rk_cap_minmax)
% 线性 AE（=PCA）主子空间基（稳健版）

    mu = mean(Y,1);
    d  = size(Y,2);
    if d == 0
        W  = zeros(0,0);
        rc = 0;
        return;
    end
    Yc = Y - mu;
    [~,S,V] = svd(Yc,'econ'); %#ok<ASGLU>
    sing = diag(S).^2;
    tot  = sum(sing);

    if tot <= eps
        rc = 1;
    else
        cs = cumsum(sing)/tot;
        idx = find(cs >= EVR_target, 1, 'first');
        if isempty(idx), idx = min(d, numel(sing)); end
        rc = idx;
    end
    rc = min(max(rc, rk_cap_minmax(1)), rk_cap_minmax(2));
    rc = min(rc, size(V,2));
    rc = max(rc, 1);

    W = V(:,1:rc);
end

%% ======================= 目标列分组 =======================

function [O_groups,K] = group_objectives_angle(PopObj, M)
% 目标列分组（稳健版）

    K = min([6, max(3, ceil(M/4)), M]);
    Y = PopObj - min(PopObj,[],1);
    cols = Y ./ max(eps,std(Y,0,1));
    cols = cols ./ max(eps, sqrt(sum(cols.^2,1)));

    % farthest-first seeds
    seeds = zeros(1,K);
    seeds(1) = randi(M);
    dist = 1 - cols(:,seeds(1))' * cols;
    for k = 2:K
        [~,seeds(k)] = max(dist);
        dist = min(dist, 1 - cols(:,seeds(k))' * cols);
    end

    % 初次赋值
    O_groups = cell(1,K);
    for j = 1:M
        [~,kk] = max( cols(:,seeds)' * cols(:,j) );
        O_groups{kk} = [O_groups{kk}, j];
    end

    % 若有空组：从当前最大组搬一个目标列过去
    emptyIdx = find(cellfun(@isempty,O_groups));
    while ~isempty(emptyIdx)
        lens = cellfun(@numel,O_groups);
        [~,big] = max(lens);
        mover = O_groups{big}(end);
        O_groups{big}(end) = [];
        O_groups{emptyIdx(1)} = mover;
        emptyIdx = find(cellfun(@isempty,O_groups));
    end
end

%% ======================= 线性逆映射 T 拟合 =======================

function map = fit_T_linear_zscore(Y, X, kappa_tar)
% 在线 z-score 域拟合线性逆映射 T（SVD 岭回归闭式解）

    muY = mean(Y,1); sigY = std(Y,0,1); sigY(sigY<1e-12)=1e-12;
    muX = mean(X,1); sigX = std(X,0,1); sigX(sigX<1e-12)=1e-12;
    Yz = (Y - muY)./sigY;
    Xz = (X - muX)./sigX;

    [U,S,V] = svd(Yz,'econ');
    s  = diag(S);
    s2 = s.^2;

    if isempty(s2)
        lambda = 1e-6;
    else
        smax2 = max(s2); smin2 = min(s2);
        if smin2==0
            lambda = smax2/(kappa_tar-1);
        else
            lambda = max(0, (smax2 - kappa_tar*smin2)/(kappa_tar-1) );
        end
        if ~isfinite(lambda) || lambda<=0
            lambda = 1e-6;
        end
    end

    if isempty(s)
        T = zeros(size(Y,2), size(X,2));
    else
        gain = s ./ (s2 + lambda);
        UX   = U' * Xz;
        T    = V * (bsxfun(@times, gain, UX));
    end

    map.T    = T;
    map.muY  = muY;  map.sigY = sigY;
    map.muX  = muX;  map.sigX = sigX;
end
