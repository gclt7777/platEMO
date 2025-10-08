
function [X0, model] = preimage_krr_map(X_all, Y_all, Y_new, opts)
% Pre-image via KRR: learn Y->X and predict X0 for new Y.
% opts: lambda, sigma, lb, ub
if nargin<4, opts = struct(); end
if ~isfield(opts,'lambda'), opts.lambda = 1e-2; end
if ~isfield(opts,'sigma'),  opts.sigma  = []; end
if ~isfield(opts,'lb'),     opts.lb     = -inf(1,size(X_all,2)); end
if ~isfield(opts,'ub'),     opts.ub     =  inf(1,size(X_all,2)); end

Kyy = rbf_kernel(Y_all, Y_all, opts.sigma);
Alpha = (Kyy + opts.lambda*eye(size(Kyy))) \ X_all;

Kny = rbf_kernel(Y_new, Y_all, opts.sigma);
X0  = Kny * Alpha;

% clamp to bounds
X0 = min(max(X0, repmat(opts.lb,size(X0,1),1)), repmat(opts.ub,size(X0,1),1));
model = struct('Alpha',Alpha,'Ytrain',Y_all,'sigma',opts.sigma,'lambda',opts.lambda);
end
