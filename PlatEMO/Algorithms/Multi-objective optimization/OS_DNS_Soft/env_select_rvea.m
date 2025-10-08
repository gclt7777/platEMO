function [PopOut,FrontNoOut,CrowdDisOut] = env_select_rvea(PopAll, N, theta)
% RVEA-like environmental selection using APD (Angle-Penalized Distance)
% - 生成 N 个参考向量 W；
% - 目标归一化，按“与参考向量的最小夹角”关联个体；
% - APD = d_parallel + theta * d_perp，按每个参考向量选择 APD 最小者；
% - 若不足 N，用全局最小 APD 补齐。
% 说明：本实现省略了代数自适应项(t/T)^alpha；可用外部 theta 近似控制惩罚强度。

    if nargin < 3 || isempty(theta), theta = 1; end

    Y = PopAll.objs;    % nAll×M
    [nAll,M] = size(Y);

    % 参考向量
    try
        [W,~] = UniformPoint(N,M);
    catch
        W = rand(N,M);
    end
    W = max(W,1e-12);
    W = W ./ sqrt(sum(W.^2,2));   % 单位化

    % 归一化目标到 [0,1]
    Zmin = min(Y,[],1); Zmax = max(Y,[],1);
    rngv = max(Zmax - Zmin, 1e-12);
    Yn   = (Y - Zmin) ./ rngv;
    Yn   = max(Yn, 0);

    % 单位化方向向量
    ynorm = sqrt(sum(Yn.^2,2)) + 1e-12;
    U = Yn ./ ynorm;             % N×M，每行单位化

    % 与参考向量的夹角（用 cos 最大等价于角度最小）
    Cosine = U * W';             % nAll×N
    Cosine = max(min(Cosine,1),-1);
    [~, assoc] = max(Cosine, [], 2);   % 每个解的关联参考向量

    % 计算 APD：沿向量距离 + theta * 垂直距离
    % d_parallel = Yn·w , d_perp = ||Yn - d_parallel * w||
    Wassoc = W(assoc,:);                 % nAll×M，每个个体对应的参考向量
    dpar   = sum(Yn .* Wassoc, 2);       % 沿向量投影
    dper   = sqrt(sum((Yn - dpar.*Wassoc).^2, 2));  % 垂距
    APD    = dpar + theta * dper;

    % 每个参考向量选 APD 最小的一个
    sel = false(nAll,1);
    for k = 1:N
        idx = find(assoc==k);
        if ~isempty(idx)
            [~,m] = min(APD(idx));
            sel(idx(m)) = true;
        end
    end

    pick = find(sel);
    if numel(pick) < N
        remain = find(~sel);
        [~,ord] = sort(APD(remain), 'ascend');
        need = N - numel(pick);
        pick = [pick; remain(ord(1:need))];
    elseif numel(pick) > N
        pick = pick(1:N);
    end

    PopOut      = PopAll(pick);
    FrontNoOut  = ones(numel(pick),1);
    CrowdDisOut = [];
end
