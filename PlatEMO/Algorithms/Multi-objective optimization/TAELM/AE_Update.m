function AE = AE_Update(Y, normState, par)
% 统一最小化 + z-score 后做 PCA，作为线性AE
% Y: N x M 目标矩阵
% 返回:
%   AE.We : M x d 编码基（右奇异向量，即主成分载荷）
%   AE.center : 1 x M 中心（在标准化空间）
%   以及用于还原量纲的 mu/sigma/signFlip

    % 方向与尺度统一
    Ymin = bsxfun(@times, Y, normState.signFlip);
    Ytil = bsxfun(@rdivide, bsxfun(@minus, Ymin, normState.mu), normState.sigma);

    % 中心化
    AE.center = mean(Ytil,1);
    Yc = bsxfun(@minus, Ytil, AE.center);

    % PCA（SVD, econ）
    [~,~,V] = svd(Yc,'econ');           % 注意用 V，不是 U
    d = min([par.latentDim, size(Y,2), size(V,2)]);
    AE.We = V(:,1:d);                    % M x d

    % 还原信息
    AE.mu       = normState.mu;
    AE.sigma    = normState.sigma;
    AE.signFlip = normState.signFlip;
end
