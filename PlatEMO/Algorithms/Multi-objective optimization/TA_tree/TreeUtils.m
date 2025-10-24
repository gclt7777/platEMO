classdef TreeUtils
% 工具集（全静态）：
% - NormalizeObjs：理想-纳迪尔归一化到 [0,1]
% - FitLinearAE / AE_Encode / AE_Decode：线性 AE（SVD）编码解码
% - SBX_PM：潜空间 SBX + 多项式变异
% - ModelTreeFit / ModelTreePredict：模型树预像（共享切分 + 叶子多输出线性回归）
% - 基础工具：pdist2 安全版、分位数

methods(Static)
    %% ---------- 归一化 ----------
    function [Yn, zmin, zmax] = NormalizeObjs(Y, zmin, zmax)
        if isempty(Y)
            Yn = []; zmin=[]; zmax=[]; return;
        end
        if isempty(zmin) || isempty(zmax)
            zmin = min(Y,[],1);
            zmax = max(Y,[],1);
        end
        rg = max(zmax - zmin, 1e-12);
        Yn = (Y - zmin)./rg;
    end

    %% ---------- 线性 AE ----------
    function [mu, W] = FitLinearAE(Yn, k)
        mu = mean(Yn,1);
        Xc = bsxfun(@minus,Yn,mu);
        [~,~,V] = svd(Xc,'econ'); %#ok<ASGLU>
        k = min([k size(V,2)]);
        W = V(:,1:k);
    end
    function Z = AE_Encode(Y, zmin, zmax, mu, W)
        Yn = TreeUtils.NormalizeObjs(Y, zmin, zmax);
        Z  = bsxfun(@minus,Yn,mu)*W;
    end
    function Yn = AE_Decode(Z, mu, W)
        Yn = bsxfun(@plus, Z*W', mu);
    end

    %% ---------- 潜空间 SBX + 多项式变异 ----------
    function Zc = SBX_PM(Z1, Z2, etaC, etaM, pm)
        [N,k] = size(Z1);
        % SBX
        u = rand(N,k);
        beta = zeros(N,k);
        idx = u <= 0.5;
        beta(idx)  = (2*u(idx)).^(1/(etaC+1));
        beta(~idx) = (2-2*u(~idx)).^(-1/(etaC+1));
        child1 = 0.5*((1+beta).*Z1 + (1-beta).*Z2);
        child2 = 0.5*((1-beta).*Z1 + (1+beta).*Z2);
        Zc = (rand(N,1)<0.5).*child1 + (rand(N,1)>=0.5).*child2;
        % PM
        mask = rand(N,k) < pm;
        if any(mask(:))
            u = rand(N,k);
            delta = zeros(N,k);
            idm = u < 0.5;
            delta(idm)  = (2*u(idm)).^(1/(etaM+1)) - 1;
            delta(~idm) = 1 - (2*(1-u(~idm))).^(1/(etaM+1));
            Zc(mask) = Zc(mask) + delta(mask);
        end
    end

    %% ===================== 模型树：训练 =====================
    function model = ModelTreeFit(Yn, X, opts)
        % Yn: N×M（目标已归一化到[0,1]），X: N×D（决策）
        if nargin<3, opts = struct(); end
        if ~isfield(opts,'maxDepth'),      opts.maxDepth = 8; end
        if ~isfield(opts,'minLeaf'),       opts.minLeaf  = 20; end
        if ~isfield(opts,'minImp'),        opts.minImp   = 1e-6; end
        if ~isfield(opts,'nThreshPerFea'), opts.nThreshPerFea = 12; end
        if ~isfield(opts,'lambdaRidge'),   opts.lambdaRidge = 1e-4; end

        [N, M] = size(Yn); %#ok<ASGLU>
        D      = size(X,2);

        % 根节点
        idxAll = (1:size(Yn,1))';
        node = TreeUtils.buildNode(Yn, X, idxAll, 0, opts, D);

        model.type  = 'ModelTree';
        model.root  = node;
        model.M     = M;  model.D = D;
        model.opts  = opts;
    end

    function Xhat = ModelTreePredict(model, Yq)
        % 逐样本下行到叶子，使用叶子的线性模型
        B = size(Yq,1); D = model.D;
        Xhat = zeros(B,D);
        for b = 1:B
            Xhat(b,:) = TreeUtils.predictOne(model.root, Yq(b,:));
        end
    end

    %% ===================== 基础工具 =====================
    function D2 = pdist2_sq(A,B)
        try
            D2 = pdist2(A,B,'squaredeuclidean');
        catch
            AA = sum(A.^2,2); BB = sum(B.^2,2).';
            D2 = bsxfun(@plus,AA,BB) - 2*(A*B.');
            D2 = max(D2,0);
        end
    end
    function D = pdist2_full(A,B)
        try
            D = pdist2(A,B);
        catch
            D = sqrt(TreeUtils.pdist2_sq(A,B));
        end
    end
    function qv = quantile_fast(x, q)
        x = x(:);
        N = numel(x);
        if N==0, qv = NaN; return; end
        q = min(max(q,0),1);
        xs = sort(x,'ascend');
        k = max(1, min(N, ceil(q*N)));
        qv = xs(k);
    end
end

%% ===================== 私有：模型树内部 =====================
methods(Static, Access=private)
    function node = buildNode(Y, X, idx, depth, opts, D)
        % 计算当前结点的 SSE（多输出）
        Xi  = X(idx,:);
        muX = mean(Xi,1);
        SSE_parent = sum(sum((Xi - muX).^2,2));

        % 停止条件
        if depth >= opts.maxDepth || numel(idx) <= 2*opts.minLeaf
            node = TreeUtils.makeLeaf(Y, X, idx, opts, D);
            return;
        end

        % 搜索最佳切分
        [bestJ, bestT, bestImp, Lidx, Ridx] = TreeUtils.findBestSplit(Y, X, idx, opts);
        if isempty(bestJ) || bestImp < opts.minImp || numel(Lidx) < opts.minLeaf || numel(Ridx) < opts.minLeaf
            node = TreeUtils.makeLeaf(Y, X, idx, opts, D);
            return;
        end

        % 递归左右子树
        left  = TreeUtils.buildNode(Y, X, Lidx, depth+1, opts, D);
        right = TreeUtils.buildNode(Y, X, Ridx, depth+1, opts, D);

        % 组装内部结点
        node.isLeaf = false;
        node.j      = bestJ;
        node.t      = bestT;
        node.left   = left;
        node.right  = right;
        node.n      = numel(idx);
        node.muX    = muX;  % 备用回退
    end

    function node = makeLeaf(Y, X, idx, opts, D)
        Yi = Y(idx,:);  Xi = X(idx,:);
        Nl = size(Yi,1);
        Phi = [ones(Nl,1), Yi];            % 截距 + 全部特征（加性线性）
        P   = size(Phi,2);
        pen = [0; ones(P-1,1)];            % 不惩罚截距
        A = Phi.'*Phi + opts.lambdaRidge*diag(pen);
        B = Phi.'*Xi;
        % 数值稳定
        try
            L = chol(A,'lower');
            Beta = L'\(L\B);
        catch
            Beta = A\B;
        end

        node.isLeaf = true;
        node.Beta   = Beta;                % (M+1)×D
        node.n      = Nl;
        node.muX    = mean(Xi,1);
    end

    function [bestJ, bestT, bestImp, Lidx, Ridx] = findBestSplit(Y, X, idx, opts)
        Ysub = Y(idx,:);  Xsub = X(idx,:);
        [Nsub, M] = size(Ysub); %#ok<ASGLU>
        % 父 SSE
        mu = mean(Xsub,1);
        SSE_parent = sum(sum((Xsub - mu).^2,2));

        bestImp = -inf; bestJ = []; bestT = [];
        Lidx = []; Ridx = [];

        for j = 1:M
            yj = Ysub(:,j);
            % 候选阈值：分位点的中点
            qs = linspace(0,1,opts.nThreshPerFea+2);
            qs = qs(2:end-1);
            cuts = arrayfun(@(q) TreeUtils.quantile_fast(yj,q), qs);
            cuts = unique(cuts(:).','stable');
            if numel(cuts) < 1, continue; end
            % 评估每个阈值
            for c = 1:numel(cuts)
                t = cuts(c);
                lid = idx(yj <= t);
                rid = idx(yj >  t);
                if numel(lid) < opts.minLeaf || numel(rid) < opts.minLeaf
                    continue;
                end
                % 子 SSE
                XL = X(lid,:);  XR = X(rid,:);
                muL = mean(XL,1); muR = mean(XR,1);
                SSE_L = sum(sum((XL - muL).^2,2));
                SSE_R = sum(sum((XR - muR).^2,2));
                imp = SSE_parent - (SSE_L + SSE_R);
                if imp > bestImp
                    bestImp = imp; bestJ = j; bestT = t;
                    Lidx = lid; Ridx = rid;
                end
            end
        end
    end

    function xhat = predictOne(node, y)
        if node.isLeaf
            xhat = [1, y] * node.Beta;
            return;
        end
        if y(node.j) <= node.t
            xhat = TreeUtils.predictOne(node.left, y);
        else
            xhat = TreeUtils.predictOne(node.right, y);
        end
    end
end
end
