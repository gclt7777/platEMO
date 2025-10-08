
function [Yz,info] = normalize_objectives(Y, opts)
if nargin < 2, opts = struct(); end
if ~isfield(opts,'maximize_idx'), opts.maximize_idx = []; end
if ~isfield(opts,'mode'), opts.mode = 'zscore'; end
if ~isfield(opts,'eps'), opts.eps = 1e-12; end

Y = double(Y);
Yflip = ones(1, size(Y,2));
Yflip(1, opts.maximize_idx) = -1;
Y_aligned = bsxfun(@times, Y, Yflip);

switch lower(opts.mode)
    case 'zscore'
        mu = mean(Y_aligned,1);
        sg = std(Y_aligned,0,1);
        sg = max(sg, opts.eps);
        Yz = bsxfun(@rdivide, bsxfun(@minus, Y_aligned, mu), sg);
        info.mode = 'zscore'; info.mu = mu; info.sg = sg;
    case 'minmax'
        lo = min(Y_aligned, [], 1);
        hi = max(Y_aligned, [], 1);
        den = max(hi - lo, opts.eps);
        Yz = bsxfun(@rdivide, bsxfun(@minus, Y_aligned, lo), den);
        info.mode = 'minmax'; info.lo = lo; info.hi = hi;
    otherwise
        error('Unknown mode: %s', opts.mode);
end
info.flip = Yflip;
end
