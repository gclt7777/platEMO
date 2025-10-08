function X_new = preimage_kernel_map(Y_target, X_ref, Y_ref, params, bounds)
% PREIMAGE_KERNEL_MAP  基于核岭回归 (KRR) 的预像映射 + 盒约束优化
%   X_new = preimage_kernel_map(Y_target, X_ref, Y_ref, params, bounds)
%   输入:
%     Y_target : K x M   目标空间的待映射点
%     X_ref    : N x D   参考决策矩阵（来自当前种群）
%     Y_ref    : N x M   对应目标矩阵
%     params   : 结构体
%                .kernel='rbf' | 'poly' (仅实现 rbf)
%                .sigma        RBF 宽度
%                .lambda       KRR 正则
%                .beta         预像正则（靠近参考点）
%                .maxIter      fmincon/PGD 最大迭代
%                .restarts     每个 y* 的随机重启次数
%                .useFmincon   是否使用 fmincon（若无许可则自动 false）
%                .verbose
%     bounds   : 结构体，.lb 1 x D, .ub 1 x D
%
%   输出:
%     X_new    : K x D  预像映射得到的决策变量
%
% 说明：
%   1) 先在参考集上拟合 KRR：Alpha = (K + lambda I) \ Y_ref
%   2) 对每个 y*，最小化 L(x) = ||k(x,X_ref)*Alpha - y*||^2 + beta||x - x_nn||^2,
%      其中 x_nn 是根据 Y_ref 离 y* 最近的参考点（初始化/正则中心）
%   3) 若有 fmincon 则用梯度优化；否则用简单的投影梯度下降 (PGD)

% ---- 预处理 ----
[N, D] = size(X_ref);
K = size(Y_target,1);
M = size(Y_ref,2);

sigma  = params.sigma;
lambda = params.lambda;
beta   = params.beta;
maxIter= params.maxIter;
restarts = max(1, params.restarts);
useFmincon = isfield(params,'useFmincon') && params.useFmincon;
verbose = isfield(params,'verbose') && params.verbose;

% ---- KRR 拟合 ----
Kmat = rbf_kernel(X_ref, X_ref, sigma);
Alpha = (Kmat + lambda*eye(N)) \ Y_ref;   % N x M

% 预计算参考 Y 到目标的最近邻（初始化与正则中心）
idxNN = knnsearch(Y_ref, Y_target);
X_center = X_ref(idxNN,:);

% ---- 对每个目标 y* 做预像优化 ----
X_new = zeros(K, D);
for k = 1:K
    ystar = Y_target(k,:);
    xc = X_center(k,:);
    bestx = xc;
    bestf = inf;

    for r = 1:restarts
        x0 = xc + 0.1*(bounds.ub - bounds.lb).*randn(1,D);
        x0 = min(max(x0, bounds.lb), bounds.ub);

        if useFmincon
            opts = optimoptions('fmincon','Display','off','SpecifyObjectiveGradient',true, ...
                'MaxIterations',maxIter,'OptimalityTolerance',1e-6,'StepTolerance',1e-8);
            [xr, fr] = fmincon(@(x)loss_with_grad(x, X_ref, Alpha, ystar, xc, sigma, beta), x0, [],[],[],[], ...
                bounds.lb, bounds.ub, [], opts);
        else
            [xr, fr] = pgd_minimize(x0, @(x)loss_with_grad(x, X_ref, Alpha, ystar, xc, sigma, beta), ...
                bounds, maxIter);
        end

        if fr < bestf
            bestf = fr; bestx = xr;
        end
    end
    X_new(k,:) = bestx;
    if verbose && mod(k, max(1,floor(K/5)))==0
        fprintf('[preimage] %d/%d bestf=%.4e\n', k, K, bestf);
    end
end

end % === 预像主函数结束 ===


% ====== 工具函数们 ======
function Kxy = rbf_kernel(X, Y, sigma)
% RBF 核: exp(-||x-y||^2 / (2*sigma^2))
X2 = sum(X.^2,2);
Y2 = sum(Y.^2,2)';
dist2 = max(bsxfun(@plus, X2, Y2) - 2*(X*Y'), 0);
Kxy = exp(-dist2 / (2*sigma^2));
end

function [f, g] = loss_with_grad(x, Xref, Alpha, ystar, xc, sigma, beta)
% f(x) = ||k(x)^T Alpha - y*||^2 + beta||x - xc||^2
% 其中 k_i(x) = exp(-||x - x_i||^2 / (2sigma^2))
x = x(:)';
N = size(Xref,1);
D = size(Xref,2);
M = size(Alpha,2);

% k(x) 及其对 x 的导数（N x 1 和 N x D）
diff = Xref - x;             % N x D
dist2 = sum(diff.^2, 2);     % N x 1
kvec  = exp(-dist2 / (2*sigma^2));  % N x 1

% yhat(x) = k(x)^T Alpha
yhat = (kvec' * Alpha);      % 1 x M
res  = yhat - ystar;         % 1 x M
f    = sum(res.^2) + beta*sum((x - xc).^2);

if nargout > 1
    % grad k_i(x) = k_i(x) * (x_i - x) / sigma^2  （注意符号）
    % J = dyhat/dx = sum_i alpha_i * grad k_i(x)  -> M x D
    % 用矩阵化： G = (Alpha' .* kvec') * (Xref - x) / sigma^2
    % 其中 Alpha' 是 M x N, (Alpha' .* kvec') 做到按样本加权
    G = (Alpha' .* kvec') * (Xref - x) / (sigma^2);  % M x D
    g = 2 * (res * G) + 2*beta*(x - xc);            % 1 x D
    g = g(:);
end
end

function [x, f] = pgd_minimize(x0, fun, bounds, maxIter)
% 简单的投影梯度下降（步长自适应 backtracking）
x = x0(:)';
[f, g] = fun(x);
alpha = 1e-1;
for it = 1:maxIter
    x_new = x - alpha * g';
    x_new = min(max(x_new, bounds.lb), bounds.ub);
    [f_new, g_new] = fun(x_new);
    if f_new <= f - 1e-4 * alpha * (g'*g)
        x = x_new; f = f_new; g = g_new;
        alpha = min(alpha * 1.1, 1); % 轻微增大
    else
        alpha = max(alpha * 0.5, 1e-6); % 回退
    end
    if norm(g) < 1e-6
        break;
    end
end
end
