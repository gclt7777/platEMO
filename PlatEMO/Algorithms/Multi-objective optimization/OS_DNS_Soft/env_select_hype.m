function [PopOut,FrontNoOut,CrowdDisOut] = env_select_hype(PopAll, N)
% Hypervolume-based environmental selection (HypE-style via Monte Carlo)
% 最小化目标；对高维推荐较大的采样 S 以提升稳定性

    Y = PopAll.objs;                % nAll × M
    [nAll,M] = size(Y);

    if nAll <= N
        PopOut      = PopAll;
        FrontNoOut  = ones(nAll,1);
        CrowdDisOut = [];
        return;
    end

    % ---------- 归一化到 [0,1] ----------
    Zmin = min(Y,[],1);
    Zmax = max(Y,[],1);
    denom = max(Zmax - Zmin, 1e-12);
    Yn = (Y - Zmin) ./ denom;                 % nAll × M
    Yn = min(max(Yn,0),1);

    % ---------- Monte Carlo 采样 ----------
    S   = max(12000, round(20*nAll + 80*M));  % 高维更稳的采样量
    ref = 1.15;                                % 参考点稍放大
    Zs = rand(S,M) * ref;                      % S × M

    % ---------- 支配矩阵 D：Zs 是否被个体 i 支配 ----------
    D = false(S,nAll);
    for i = 1:nAll
        D(:,i) = all( bsxfun(@ge, Zs, Yn(i,:)), 2 );  % 最小化：点 >= y 即被 y 支配
    end

    if ~any(any(D,2))
        [PopOut,FrontNoOut,CrowdDisOut] = fallback_cheby(PopAll,N,Yn);
        return;
    end

    % ---------- 迭代移除贡献最小的个体 ----------
    alive = true(nAll,1);  curN = nAll;

    while curN > N
        idx  = find(alive);
        Di   = D(:,idx);                 % S × curN
        Ci   = sum(Di,2);                % 每个样本被多少个存活解支配
        mask1 = (Ci==1);                 % 仅被一个体支配的样本
        uniq = zeros(numel(idx),1);
        if any(mask1)
            D1 = Di(mask1,:);            % (#uniqSamples) × curN
            uniq = sum(D1,1)';           % 每个体的“唯一贡献”
        end

        if all(uniq==0)                  % 二级准则：最小切比雪夫值优先保留
            gbest = max(Yn(idx,:),[],2);
            [~,ord] = sort(gbest,'ascend');
            remove_local = ord(end);
        else
            [~,ord] = sort(uniq,'ascend');
            remove_local = ord(1);
        end

        alive(idx(remove_local)) = false;
        curN = curN - 1;
    end

    sel = find(alive);
    PopOut      = PopAll(sel);
    FrontNoOut  = ones(numel(sel),1);
    CrowdDisOut = [];
end

% ---------- 简单 Cheby 兜底 ----------
function [PopOut,FrontNoOut,CrowdDisOut] = fallback_cheby(PopAll,N,Yn)
    g = max(Yn,[],2);
    [~,ord] = sort(g,'ascend');
    sel = ord(1:min(N,numel(ord)));
    PopOut      = PopAll(sel);
    FrontNoOut  = ones(numel(sel),1);
    CrowdDisOut = [];
end
