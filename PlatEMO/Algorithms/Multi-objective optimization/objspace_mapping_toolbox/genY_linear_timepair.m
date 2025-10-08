
function [Y_new, model] = genY_linear_timepair(Y_src, Y_tar, Y_curr, opts)
if nargin < 4, opts = struct(); end
if ~isfield(opts,'alpha'),  opts.alpha  = 0.6; end
if ~isfield(opts,'lambda'), opts.lambda = 1e-3; end
if ~isfield(opts,'norm'),   opts.norm   = struct(); end

[YS, info] = normalize_objectives(Y_src, opts.norm);
YT = normalize_objectives(Y_tar, opts.norm);
YC = normalize_objectives(Y_curr, opts.norm);

X = [YS, ones(size(YS,1),1)];
M = size(YS,2);
Reg = diag([ones(M,1)*opts.lambda; 0]);
A = (X' * X + Reg) \ (X' * YT);
W = A(1:end-1,:); b = A(end,:);

Y_pred = YC * W + b;
Y_mix  = (1-opts.alpha)*YC + opts.alpha*Y_pred;
Y_new  = unnormalize_objectives(Y_mix, info);

model = struct('W',W,'b',b,'info',info,'opts',opts);
end
