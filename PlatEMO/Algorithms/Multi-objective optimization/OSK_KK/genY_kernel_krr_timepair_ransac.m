
function [Y_new, model] = genY_kernel_krr_timepair_ransac(Y_src, Y_tar, Y_curr, opts)
% Robust KRR map in Y-space: train on RANSAC-selected subset, then refit on inliers.
if nargin < 4, opts = struct(); end
if ~isfield(opts,'lambda'), opts.lambda = 1e-2; end
if ~isfield(opts,'alpha'),  opts.alpha  = 0.6; end
if ~isfield(opts,'sigma'),  opts.sigma  = []; end
if ~isfield(opts,'norm'),   opts.norm   = struct(); end
if ~isfield(opts,'maxIter'),opts.maxIter= 25; end
if ~isfield(opts,'subRatio'),opts.subRatio=0.7; end

[YS, info] = normalize_objectives(Y_src, opts.norm);
YT = normalize_objectives(Y_tar, opts.norm);
YC = normalize_objectives(Y_curr, opts.norm);

N = size(YS,1);
subN = max(10, floor(opts.subRatio*N));
bestMed = inf; bestAlpha = []; bestTrain = [];

for t = 1:opts.maxIter
    idx = randperm(N, subN);
    Kss = rbf_kernel(YS(idx,:), YS(idx,:), opts.sigma);
    Alpha0 = (Kss + opts.lambda*eye(subN)) \ YT(idx,:);

    % evaluate residuals on ALL using this subset model
    Kall = rbf_kernel(YS, YS(idx,:), opts.sigma);
    Yhat = Kall * Alpha0;
    R    = Yhat - YT;
    rnorm = sqrt(sum(R.^2,2));
    medr = median(rnorm);
    if medr < bestMed
        bestMed = medr; bestAlpha = Alpha0; bestTrain = idx;
    end
end

% inliers & refit full KRR on them
Kall = rbf_kernel(YS, YS(bestTrain,:), opts.sigma);
Yhat = Kall * bestAlpha;
rnorm = sqrt(sum((Yhat - YT).^2,2));
thr = 1.4826*median(rnorm)*2;
inl = rnorm <= max(thr,1e-8);

K = rbf_kernel(YS(inl,:), YS(inl,:), opts.sigma);
Alpha = (K + opts.lambda*eye(sum(inl))) \ YT(inl,:);

Kc = rbf_kernel(YC, YS(inl,:), opts.sigma);
Y_pred = Kc * Alpha;
Y_mix  = (1-opts.alpha)*YC + opts.alpha*Y_pred;
Y_new  = unnormalize_objectives(Y_mix, info);

model = struct('Alpha',Alpha,'Ytrain',YS(inl,:),'info',info,'sigma',opts.sigma,'lambda',opts.lambda,'alpha',opts.alpha);
end
