
function X_new = preimage_refine_fmincon(X0, Y_targets, X_train, Y_train, opts)
if nargin < 5, opts = struct(); end
if ~isfield(opts,'lambda'),  opts.lambda  = 1e-3; end
if ~isfield(opts,'sigmaX'),  opts.sigmaX  = []; end
if ~isfield(opts,'maxIter'), opts.maxIter = 100; end
assert(isfield(opts,'lb') && isfield(opts,'ub'), 'Provide opts.lb and opts.ub for box constraints.');

Kxx = rbf_kernel(X_train, X_train, opts.sigmaX);
N = size(X_train,1);
BetaXY = (Kxx + opts.lambda * eye(N)) \ Y_train;

opt = optimoptions('fmincon','Display','none','Algorithm','interior-point','MaxIterations',opts.maxIter);

Nt = size(Y_targets,1);
X_new = zeros(Nt, size(X_train,2));

    function yhat = fhat(xrow)
        k = rbf_kernel(xrow, X_train, opts.sigmaX);
        yhat = k * BetaXY;
    end

for i = 1:Nt
    ystar = Y_targets(i,:);
    x0    = X0(i,:);
    fun = @(x) sum((fhat(x) - ystar).^2);
    X_new(i,:) = fmincon(fun, x0, [],[],[],[], opts.lb, opts.ub, [], opt);
end
end
