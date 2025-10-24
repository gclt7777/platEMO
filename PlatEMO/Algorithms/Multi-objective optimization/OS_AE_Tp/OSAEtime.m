function OSAEtime(Global)
% <algorithm> <L>
% OS_AE_T : Objective-space Grouping + AE(PCA) + Linear T write-back
%
% - 目标聚类 → 唯一指派变量列 S_k（按 T 子块能量）
% - 组内：PCA 潜空间 + DE(rand/1/bin) 生成新的目标向量
% - 写回：仅改各自 S_k 列，半步融合 eta
% - T 为 zscore 域的岭回归 SVD 解；支持 EMA 平滑
% - periodGroup：重分组/重指派周期；tPeriod：T 的拟合周期（独立于分组）

    %% 参数
    %  K,topFrac,lambda,eta,kcap,kfrac,F,Cr,periodGroup,emaA,tPeriod
    [K,topFrac,lambda,eta,kcap,kfrac,F,Cr,periodGroup,emaA,tPeriod] = ...
        Global.ParameterSet(3,0.2,1e-5,0.5,4,1/3,0.7,0.9,5,0.0,1);

    %% 初始化
    Population = Global.Initialization();
    gen      = 1;
    map      = [];
    map_prev = [];

    [~,D] = size(Population.decs);

    % 首代：分组 + 拟合 T + 指派
    O_groups  = group_by_objective(Population,K);
    map       = fitMap_ridgeSVD(Population.objs,Population.decs,lambda);
    if emaA>0, map_prev = map; end
    S_groups  = assign_S_groups(map,O_groups,topFrac,D);

    %% 主循环
    while Global.NotTermination(Population)
        X = Population.decs;   % N×D
        Y = Population.objs;   % N×M
        OffDec = X;

        % —— 节奏控制：分组 与 T 拟合（互相独立） ——
        doReGroup = (gen==1) || (mod(gen-1,periodGroup)==0);
        doFitT    = (gen==1) || (mod(gen-1,tPeriod)==0);

        % —— 分组（仅按 Y），不动 T ——
        if doReGroup
            O_groups = group_by_objective(Population,K);
        end

        % —— 拟合/更新 T（可 EMA 平滑） ——
        if doFitT
            map_new = fitMap_ridgeSVD(Y,X,lambda);
            if emaA>0 && ~isempty(map_prev)
                map_blend.T   = (1-emaA)*map_prev.T + emaA*map_new.T;
                map_blend.muY = map_new.muY;  map_blend.sigY = map_new.sigY;
                map_blend.muX = map_new.muX;  map_blend.sigX = map_new.sigX;
                map = map_blend; map_prev = map_blend;
            else
                map = map_new;  map_prev = map_new;
            end
        end

        % —— 若分组或 T 变了，则重算唯一指派 S_groups —— 
        if doReGroup || doFitT || isempty(S_groups)
            S_groups = assign_S_groups(map,O_groups,topFrac,D);
        end

        % —— 分组：PCA→DE 生成，并用子块 T(Ok,Sk) 写回各自列 ——
        for k = 1:numel(O_groups)
            Ok = O_groups{k};  Sk = S_groups{k};
            if isempty(Ok) || isempty(Sk), continue; end

            Yk      = Y(:,Ok);
            Yk_new  = ae_pca_generate(Yk,kcap,kfrac,F,Cr);      % N×|Ok|
            Xhat_Sk = apply_map_sub(map,Yk_new,Ok,Sk);           % 反标准化写回

            % 半步融合
            OffDec(:,Sk) = (1-eta)*OffDec(:,Sk) + eta*Xhat_Sk;
        end

        % 边界裁剪 + 评估 + 环境选择（非支配+角度截断）
        OffDec    = min(max(OffDec,Global.lower),Global.upper);
        Offspring = INDIVIDUAL(OffDec);
        Population = env_select_local([Population,Offspring], Global.N);

        gen = gen + 1;
    end
end

%% ============ 目标分组 / 唯一指派 ============

function O_groups = group_by_objective(Population,K)
    Y = Population.objs; M = size(Y,2);
    if isempty(K) || ~isscalar(K) || K<1, K = min(3,M); end
    if M==1, K_eff=1; else, K_eff = min(K, max(1,M-1)); end
    try
        [Og,~,~] = OS_GroupByObjective(Population,'K',K_eff);
        O_groups = Og;
    catch
        O_groups = os_group_by_objective_builtin(Y,K_eff);
    end
end

function S_groups = assign_S_groups(map,O_groups,topFrac,D)
    % 稳健：topFrac∈[0,1]；输出统一为列向量（空为 0×1）
    topFracEff = min(max(topFrac,0),1);
    Keff = numel(O_groups);
    energy = zeros(Keff,D);
    for k = 1:Keff
        Ok = O_groups{k};
        if isempty(Ok), continue; end
        Tk = map.T(Ok,:);                    % |Ok|×D
        energy(k,:) = sqrt(sum(Tk.^2,1));    % 子块能量
    end
    [~,owner] = max(energy,[],1);            % 每列归属组

    S_groups = cell(1,Keff);
    for k = 1:Keff
        idx = find(owner==k);
        cnt = numel(idx);
        if cnt==0
            S_groups{k} = zeros(0,1);
            continue;
        end
        e = energy(k,idx); e(~isfinite(e)) = -inf;
        [~,ord] = sort(e,'descend');
        topK = max(1, min(cnt, round(topFracEff*cnt)));
        S_groups{k} = reshape(idx(ord(1:topK)),[],1); % 强制列向量
    end
end

%% ============ 拟合/生成 ============

function map = fitMap_ridgeSVD(Y,X,lambda)
    % 在标准化域解 T，并保存反标准化所需统计量
    muY = mean(Y,1);  sigY = std(Y,0,1); sigY(sigY==0)=1;
    muX = mean(X,1);  sigX = std(X,0,1); sigX(sigX==0)=1;

    Yz = (Y - muY)./sigY;
    [U,S,V] = svd(Yz,'econ'); sig = diag(S);
    G  = V * diag(sig./(sig.^2 + lambda)) * U';   % (Y'Y+λI)^{-1}Y'
    Xz = (X - muX)./sigX;
    T  = G * Xz;                                   % M×D

    map.T=T; map.muY=muY; map.sigY=sigY; map.muX=muX; map.sigX=sigX;
end

function Y_new = ae_pca_generate(Yk,kcap,kfrac,F,Cr)
    [Ykz,mu,sg] = safe_zscore(Yk);
    r  = max(1, min([size(Yk,2), max(1, ceil(size(Yk,2)*kfrac)), kcap]));
    W  = pca_basis(Ykz,r);
    Z  = Ykz * W;

    % 一步 DE(rand/1/bin) 于潜空间
    [N,k] = size(Z); Znew = Z;
    if N>=4 && k>0
        idx = 1:N;
        for i=1:N
            rset = idx; rset(i)=[];
            rset = rset(randperm(numel(rset),3));
            v = Z(rset(1),:) + F*(Z(rset(2),:)-Z(rset(3),:));
            jrand = randi(k);
            mask = (rand(1,k) < Cr); mask(jrand) = true;
            u = Z(i,:); u(mask) = v(mask);
            Znew(i,:) = u;
        end
    end
    Y_new = (Znew * W') .* sg + mu;
end

function [Z,mu,sg] = safe_zscore(X)
    mu = mean(X,1);
    sg = std(X,0,1); sg(sg==0)=1;
    Z  = (X - mu) ./ sg;
end

function W = pca_basis(X,k)
    [~,~,V] = svd(X,'econ'); W = V(:,1:k);
end

function Xhat_Sk = apply_map_sub(map,Yk_new,Ok,Sk)
    Yz  = bsxfun(@rdivide, bsxfun(@minus, Yk_new, map.muY(Ok)), map.sigY(Ok));
    Xz  = Yz * map.T(Ok,Sk);
    Xhat_Sk = bsxfun(@plus, bsxfun(@times, Xz, map.sigX(Sk)), map.muX(Sk));
end

%% ============ 环境选择（非支配 + 角度截断） ============

function Population = env_select_local(PopBoth,N)
    PopObj = PopBoth.objs;
    [FrontNo, MaxFNo] = NDSort(PopObj, N);
    Next = FrontNo < MaxFNo;
    Last = find(FrontNo == MaxFNo);
    K = N - sum(Next);
    if K > 0
        Choose = truncation_angle(PopObj(Last,:), K);
        Next(Last(Choose)) = true;
    end
    Population = PopBoth(Next);
end

function Choose = truncation_angle(PopObj,K)
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

%% ============ 内置的目标聚类兜底 ============

function O_groups = os_group_by_objective_builtin(Y,K)
    [~,M] = size(Y);
    if M==1 || K<=1, O_groups = {1:M}; return; end
    K = min(K, max(1,M-1));

    Yz = zscore(Y,0,1);
    Yz(~isfinite(Yz)) = 0;               % 轻量防护
    nrm = sqrt(sum(Yz.^2,1)); nrm(nrm==0)=1;
    U = Yz ./ nrm;
    S = U.'*U; S(1:M+1:end)=0;

    A = max(S,0); A = A - diag(diag(A));
    d = sum(A,2); Dm = diag(d + eps);
    Lsym = eye(M) - Dm^(-1/2)*A*Dm^(-1/2); Lsym = (Lsym+Lsym.')/2;

    [V,E] = eig(Lsym);
    [~,ord] = sort(diag(E),'ascend');
    H  = V(:,ord(1:K));
    Hn = H ./ max(sqrt(sum(H.^2,2)),eps); Hn(~isfinite(Hn))=0;

    try
        opts = statset('MaxIter',200,'Display','off');
        repl = max(5, min(10, M-1));
        lbl  = kmeans(Hn,K,'Replicates',repl,'Options',opts);
    catch
        % 无统计工具箱时兜底：按顺序均匀分配
        idx = 1:M; lbl = mod(idx-1,K)+1;
    end
    O_groups = cell(1,K);
    for k = 1:K, O_groups{k} = find(lbl==k).'; end
end
