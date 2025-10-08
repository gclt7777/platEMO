
function X_new = objective_space_linear_kernel_preimage(X, Y, histY, histX, params)
% Wrapper: generate Y_new (linear or kernel; robust or not) -> pre-image -> optional refine.
arguments
    X double
    Y double
    histY double
    histX double
    params.mode         (1,:) char   = 'linear'   % 'linear' or 'kernel'
    params.alpha        (1,1) double = 0.6
    params.lambdaY      (1,1) double = 1e-3
    params.sigmaY       = []
    params.norm         struct       = struct()
    params.lambdaInv    (1,1) double = 1e-3
    params.sigmaInv     = []
    params.lambdaF      (1,1) double = 1e-3
    params.sigmaF       = []
    params.refineIters  (1,1) double = 60
    params.lb           double
    params.ub           double
    params.robust       logical = false
end

% 1) Generate new Y
if params.robust
    switch lower(params.mode)
        case 'linear'
            [Ynew, ~] = genY_linear_timepair_ransac(histY, Y, Y, struct('alpha',params.alpha, 'lambda',params.lambdaY, 'norm',params.norm));
        case 'kernel'
            [Ynew, ~] = genY_kernel_krr_timepair_ransac(histY, Y, Y, struct('alpha',params.alpha, 'lambda',params.lambdaY, 'sigma',params.sigmaY, 'norm',params.norm));
        otherwise
            error('Unknown params.mode: %s', params.mode);
    end
else
    switch lower(params.mode)
        case 'linear'
            [Ynew, ~] = genY_linear_timepair(histY, Y, Y, struct('alpha',params.alpha, 'lambda',params.lambdaY, 'norm',params.norm));
        case 'kernel'
            [Ynew, ~] = genY_kernel_krr_timepair(histY, Y, Y, struct('alpha',params.alpha, 'lambda',params.lambdaY, 'sigma',params.sigmaY, 'norm',params.norm));
        otherwise
            error('Unknown params.mode: %s', params.mode);
    end
end

% 2) Pre-image: Y->X
[X0, ~] = preimage_krr_map([histX; X], [histY; Y], Ynew, struct('sigma',params.sigmaInv,'lambda',params.lambdaInv,'lb',params.lb,'ub',params.ub));

% 3) Optional refine with forward KRR + fmincon
if params.refineIters > 0
    X_new = preimage_refine_fmincon(X0, Ynew, [histX; X], [histY; Y], struct('sigmaX',params.sigmaF,'lambda',params.lambdaF,'lb',params.lb,'ub',params.ub,'maxIter',params.refineIters));
else
    X_new = X0;
end
end
