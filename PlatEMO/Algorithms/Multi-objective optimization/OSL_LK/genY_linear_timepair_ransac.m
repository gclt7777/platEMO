
function [Y_new, model] = genY_linear_timepair_ransac(Y_src, Y_tar, Y_curr, opts)
% Robust affine map in Y-space using simple RANSAC then refit on inliers.
if nargin < 4, opts = struct(); end
if ~isfield(opts,'alpha'),  opts.alpha  = 0.6; end
if ~isfield(opts,'lambda'), opts.lambda = 1e-2; end
if ~isfield(opts,'norm'),   opts.norm   = struct(); end
if ~isfield(opts,'maxIter'),opts.maxIter= 30; end
if ~isfield(opts,'subRatio'),opts.subRatio=0.7; end

[YS, info] = normalize_objectives(Y_src, opts.norm);
YT = normalize_objectives(Y_tar, opts.norm);
YC = normalize_objectives(Y_curr, opts.norm);

N = size(YS,1); M = size(YS,2);
bestMed = inf; bestW = []; bestb = [];

subN = max(3, floor(opts.subRatio*N));
for t = 1:opts.maxIter
    idx = randperm(N, subN);
    Xsub = [YS(idx,:), ones(subN,1)];
    Reg  = diag([ones(M,1)*opts.lambda; 0]);
    A    = (Xsub' * Xsub + Reg) \ (Xsub' * YT(idx,:));
    W0   = A(1:end-1,:); b0 = A(end,:);

    R = (YS*W0 + b0) - YT;                  % residuals on ALL points
    rnorm = sqrt(sum(R.^2,2));
    medr  = median(rnorm);
    if medr < bestMed
        bestMed = medr; bestW = W0; bestb = b0;
    end
end

% refine on inliers of the best model
R = (YS*bestW + bestb) - YT;
rnorm = sqrt(sum(R.^2,2));
thr = 1.4826*median(rnorm)*2; % MAD-based
inl = rnorm <= max(thr, 1e-8);
X = [YS(inl,:), ones(sum(inl),1)];
Reg  = diag([ones(M,1)*opts.lambda; 0]);
A    = (X' * X + Reg) \ (X' * YT(inl,:));
W = A(1:end-1,:); b = A(end,:);

Y_pred = YC * W + b;
Y_mix  = (1-opts.alpha)*YC + opts.alpha*Y_pred;
Y_new  = unnormalize_objectives(Y_mix, info);

model = struct('W',W,'b',b,'info',info,'opts',opts);
end
