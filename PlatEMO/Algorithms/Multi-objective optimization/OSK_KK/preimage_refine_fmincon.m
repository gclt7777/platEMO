
function X_ref = preimage_refine_fmincon(X0, Ystar, Xtrain, Ytrain, opts)
% Refine pre-image using forward KRR surrogate + bounded local search.
% opts: lambda (for KRR), sigmaX, lb, ub, maxIter
if nargin<5, opts = struct(); end
if ~isfield(opts,'lambda'),  opts.lambda = 1e-2; end
if ~isfield(opts,'sigmaX'),  opts.sigmaX = []; end
if ~isfield(opts,'lb'),      opts.lb     = -inf(1,size(Xtrain,2)); end
if ~isfield(opts,'ub'),      opts.ub     =  inf(1,size(Xtrain,2)); end
if ~isfield(opts,'maxIter'), opts.maxIter= 60; end

% Train forward KRR: X->Y
Kxx  = rbf_kernel(Xtrain, Xtrain, opts.sigmaX);
Beta = (Kxx + opts.lambda*eye(size(Kxx))) \ Ytrain;
predictY = @(x) rbf_kernel(x, Xtrain, opts.sigmaX) * Beta;

if exist('fmincon','file') ~= 2
    % Fallback: return X0 if fmincon is unavailable
    X_ref = X0;
    return;
end

n = size(X0,1);
X_ref = X0;
options = optimoptions('fmincon','Display','off','Algorithm','interior-point',...
                       'MaxIterations',opts.maxIter,'SpecifyObjectiveGradient',false);

for i = 1:n
    ytar = Ystar(i,:);
    fun = @(x) obj_fun(x, ytar, predictY);
    try
        X_ref(i,:) = fmincon(fun, X0(i,:), [],[],[],[], opts.lb, opts.ub, [], options);
    catch
        % if optimization fails, keep X0
        X_ref(i,:) = X0(i,:);
    end
end
end

function [fval, grad] = obj_fun(x, ytar, predictY)
yh = predictY(x);
r  = yh - ytar;
fval = sum(r.^2,2);
if nargout>1
    grad = []; % numeric gradient by fmincon
end
end
