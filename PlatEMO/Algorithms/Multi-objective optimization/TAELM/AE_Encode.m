function Z = AE_Encode(Y, AE)
% Y -> 潜空间 Z
% AE: 由 AE_Update 返回的结构体
    if isempty(Y)
        Z = zeros(0, size(AE.We,2));
        return;
    end
    Ymin = bsxfun(@times, Y, AE.signFlip);
    Ytil = bsxfun(@rdivide, bsxfun(@minus, Ymin, AE.mu), AE.sigma);
    Yc   = bsxfun(@minus, Ytil, AE.center);
    Z    = Yc * AE.We;
end
