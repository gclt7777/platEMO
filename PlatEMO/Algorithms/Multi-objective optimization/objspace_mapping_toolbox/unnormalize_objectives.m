
function Y = unnormalize_objectives(Yz, info)
Yz = double(Yz);
switch lower(info.mode)
    case 'zscore'
        Y_aligned = bsxfun(@plus, bsxfun(@times, Yz, info.sg), info.mu);
    case 'minmax'
        den = max(info.hi - info.lo, 1e-12);
        Y_aligned = bsxfun(@plus, bsxfun(@times, Yz, den), info.lo);
    otherwise
        error('Unknown mode in info.mode');
end
Y = bsxfun(@times, Y_aligned, info.flip);
end
