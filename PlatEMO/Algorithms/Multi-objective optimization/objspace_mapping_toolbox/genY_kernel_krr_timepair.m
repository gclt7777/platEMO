
function [Y_new, model] = genY_kernel_krr_timepair(Y_src, Y_tar, Y_curr, opts)
if nargin < 4, opts = struct(); end
if ~isfield(opts,'lambda'), opts.lambda = 1e-3; end
if ~isfield(opts,'alpha'),  opts.alpha  = 0.6; end
if ~isfield(opts,'sigma'),  opts.sigma  = []; end
if ~isfield(opts,'norm'),   opts.norm   = struct(); end

[YS, info] = normalize_objectives(Y_src, opts.norm);
YT = normalize_objectives(Y_tar, opts.norm);
YC = normalize_objectives(Y_curr, opts.norm);

K = rbf_kernel(YS, YS, opts.sigma);
N = size(YS,1);
Alpha = (K + opts.lambda * eye(N)) \ YT;

Kc = rbf_kernel(YC, YS, opts.sigma);
Y_pred = Kc * Alpha;
Y_mix  = (1-opts.alpha)*YC + opts.alpha*Y_pred;
Y_new  = unnormalize_objectives(Y_mix, info);

model = struct('Alpha',Alpha,'Ytrain',YS,'info',info,'sigma',opts.sigma,'lambda',opts.lambda,'alpha',opts.alpha);
end
