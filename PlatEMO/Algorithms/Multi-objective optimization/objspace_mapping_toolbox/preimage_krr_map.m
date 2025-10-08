
function [X0, invModel] = preimage_krr_map(X_train, Y_train, Y_targets, opts)
if nargin < 4, opts = struct(); end
if ~isfield(opts,'lambda'), opts.lambda = 1e-3; end
if ~isfield(opts,'sigma'),  opts.sigma  = []; end

Y = double(Y_train); X = double(X_train); T = double(Y_targets);

Kyy = rbf_kernel(Y, Y, opts.sigma);
N = size(Y,1);
AlphaYX = (Kyy + opts.lambda * eye(N)) \ X;

Kty = rbf_kernel(T, Y, opts.sigma);
X0 = Kty * AlphaYX;

if isfield(opts,'lb') && isfield(opts,'ub') && ~isempty(opts.lb)
    X0 = max(bsxfun(@plus, opts.lb, zeros(size(X0))), X0);
    X0 = min(bsxfun(@plus, opts.ub, zeros(size(X0))), X0);
end

invModel = struct('AlphaYX',AlphaYX,'Ytrain',Y,'sigma',opts.sigma,'lambda',opts.lambda);
end
