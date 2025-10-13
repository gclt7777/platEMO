function x_hat = LLR_InverseMap(y_hat, MemY, MemX, normState, k, lambda, lb, ub, sectorAware)
% 局部加权岭回归（只惩罚线性项），回退机制：加权平均/最近邻
% y_hat: 1xM, MemY: NxM, MemX: NxD

    if nargin<9, sectorAware=false; end
    if isempty(MemY)
        error('Memory empty.');
    end

    % 统一最小化与标准化用于距离（不改变回归空间的量纲）
    [y_til, y_min] = toMinAndStd(y_hat, normState);
    [Y_til, Y_min] = toMinAndStd(MemY, normState);

    % KNN 邻域（此处用欧氏距离；如需扇区优先，可先筛子集再 knn）
    k = min(k, size(Y_til,1));
    D = pdist2(y_til, Y_til, 'euclidean');
    [ds, idx] = mink(D, k);

    YN = Y_min(idx,:); XN = MemX(idx,:);
    sig = median(ds); if sig<=0, sig=1e-6; end
    w = exp(-(ds.^2)/(2*sig^2)); w = w/(sum(w)+eps);

    % 加权岭回归（只惩罚线性项）
    Z = [ones(k,1), YN];                      % (k, 1+M)
    W = diag(w);
    Mdim = size(Z,2);
    Lambda = zeros(Mdim); Lambda(2:end,2:end) = lambda * eye(Mdim-1);

    A = Z' * W * Z + Lambda;
    B = Z' * W * XN;

    try
        coef = A \ B;                         % (1+M, D)
        x_hat = [1, y_min] * coef;           % 1xD
        if any(~isfinite(x_hat)), error('Ill-conditioned'); end
    catch
        % 回退：加权平均
        x_hat = w * XN;
    end

    % 变量边界修复
    if ~isempty(lb), x_hat = max(x_hat, lb); end
    if ~isempty(ub), x_hat = min(x_hat, ub); end
end

% ---------- 小工具 ----------
function [Y_til, Y_min] = toMinAndStd(Y, normState)
    Y_min = bsxfun(@times, Y, normState.signFlip);
    Y_til = bsxfun(@rdivide, bsxfun(@minus, Y_min, normState.mu), normState.sigma);
end
