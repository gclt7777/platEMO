function [Ystar] = soft_teacher_targets(Y, epsRate, kNeighbor, sigmaScale, tauCheby, useRef)
% 构造每个体的“软教师目标” Ystar（N×M）
% 1) 优先用 ε-支配的更优集合作为老师；为空则用 KNN 兜底
% 2) 老师权重 = 高斯邻近权 * Cheby 方向偏好（可选）
% 3) 返回加权均值作为 y_i^*
%
% 参数：
%   Y          : N×M 目标矩阵（最小化）
%   epsRate    : ε 相对范围比例（如 0.005~0.02）
%   kNeighbor  : 近邻个数（兜底或候选截断）
%   sigmaScale : 高斯 σ 的系数（基于中位近邻距离）
%   tauCheby   : Cheby 权重温度（0.3~0.7）
%   useRef     : 是否使用参考向量偏好（0/1）

    [N,M] = size(Y);
    Zmin = min(Y,[],1);
    Zmax = max(Y,[],1);
    rngv = max(Zmax - Zmin, 1e-12);
    Yn   = (Y - Zmin) ./ rngv;                 % 0..1 归一化
    epsv = epsRate * rngv;                     % 各维 ε

    % --- 预计算全局近邻尺度 σ ---
    Dist = pdist2(Yn, Yn, 'euclidean');        % N×N 目标空间距离
    Dist(1:N+1:end) = inf;                     % 自身距离置 inf，便于 KNN
    sorted = sort(Dist, 2, 'ascend');
    kth = sorted(:, min(5, max(1,N-1)));       % 第5近邻距离（或最接近）
    sigma = sigmaScale * median(kth(kth<inf));
    sigma = max(sigma, 1e-6);

    % --- 可选：参考向量（用于 Cheby 偏好） ---
    Wref = [];
    if useRef
        H = max(N, 2*M);   % 参考向量数量
        try
            [Wref,~] = UniformPoint(H, M);
        catch
            Wref = rand(H,M);
        end
        Wref = max(Wref, 1e-12);
        Wref = Wref ./ max(sum(Wref,2), 1e-12);
    end

    Ystar = zeros(N,M);

    for i = 1:N
        yi  = Y(i,:);
        yni = Yn(i,:);

        % ε-支配老师集合：a ε-dominates b
        % ∀k: a_k <= b_k + eps_k 且 ∃k: a_k < b_k - eps_k
        leAll = all(bsxfun(@le, Y, bsxfun(@plus, yi, epsv)), 2);
        ltAny = any(bsxfun(@lt, Y, bsxfun(@minus, yi, epsv)), 2);
        mask  = leAll & ltAny;
        mask(i) = false;

        cand = find(mask);
        if isempty(cand)
            % 兜底：KNN（按目标空间距离，利用预计算 Dist）
            [~,ord] = sort(Dist(i,:),'ascend');
            cand = ord(1:min(kNeighbor, N-1));
        else
            % 候选太多时，按距离截断至 K 近邻
            [~,ordLoc] = sort(Dist(i,cand),'ascend');
            cand = cand(ordLoc(1:min(kNeighbor, numel(cand))));
        end

        % 高斯邻近权
        d  = Dist(i,cand)';
        w  = exp(- (d.^2) / (2*sigma^2));  % 近者权重大

        % Cheby 方向偏好（小 g_te 更好）
        if ~isempty(Wref) && tauCheby > 0
            % 用 zmin 作为参考点，基于归一化 Yn 计算
            % g_te(y|w) = max_j w_j * y_j  （zmin=0 after normalization）
            g = zeros(numel(cand),1);
            for t = 1:numel(cand)
                yj = Yn(cand(t),:);
                % 取在所有参考向量下的最小 g 值（最贴近某方向）
                gt = max(bsxfun(@times, Wref, yj), [], 2);
                g(t) = min(gt);
            end
            w = w .* exp(- g / max(tauCheby,1e-6));
        end

        if all(w<=0) || isempty(cand)
            Ystar(i,:) = yi;  % 无法学习则保持不变
        else
            w = w / sum(w);
            Ystar(i,:) = w' * Y(cand,:);
        end
    end
end
