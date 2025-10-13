function Yhat = AE_Decode(Z, AE)
% 潜空间 Z -> 目标空间（还原尺度与方向）
    Yc   = Z * AE.We';
    Ytil = bsxfun(@plus, Yc, AE.center);
    Ymin = bsxfun(@plus, bsxfun(@times, Ytil, AE.sigma), AE.mu);
    Yhat = bsxfun(@times, Ymin, AE.signFlip); % 还原目标方向（最小化/最大化）
end
