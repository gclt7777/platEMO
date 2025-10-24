classdef GAMUtils
% 工具集（全静态）：
% - NormalizeObjs：理想-纳迪尔归一化到 [0,1]
% - FitLinearAE / AE_Encode / AE_Decode：线性 AE（SVD）编码解码
% - SBX_PM：潜空间 SBX + 多项式变异
% - GAMPreimageFit / GAMPreimagePredict：分段线性样条的 GAM 预像（多输出）
% - 基础工具：pdist2 安全版、分位数、单纯形投影(备用)

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
        Yn = GAMUtils.NormalizeObjs(Y, zmin, zmax);
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

    %% ---------- GAM 预像：训练 ----------
    function model = GAMPreimageFit(Yn, X, opts)
        % Yn: N×M（目标已归一化到[0,1]），X: N×D（决策）
        if nargin<3, opts = struct(); end
        if ~isfield(opts,'numKnots'),    opts.numKnots = 5; end
        if ~isfield(opts,'lambdaRidge'), opts.lambdaRidge = 1e-4; end
        if ~isfield(opts,'knotRule'),    opts.knotRule = 'quantile'; end % or 'uniform'

        [N, M] = size(Yn);
        D      = size(X,2);

        % 为每个目标维度构造“线性样条”基：phi = [1, y, max(0,y-t1),...,max(0,y-tK)]
        Phi_parts = cell(1,M);
        knots = cell(1,M);
        for j = 1:M
            yj = Yn(:,j);
            if strcmpi(opts.knotRule,'uniform')
                % 内部结点（不含 0/1）
                t = linspace(0,1,opts.numKnots+2);
                t = t(2:end-1);
            else
                % 分位点（去重）
                qs = linspace(0,1,opts.numKnots+2);
                qs = qs(2:end-1);
                t  = arrayfun(@(qq) GAMUtils.quantile_fast(yj,qq), qs);
                t  = unique(t(:).'); % 去重
            end
            knots{j} = t(:).';
            % 基： [ y , (y - t_k)_+ ]
            H = max(0, yj - t);
            Phi_parts{j} = [ yj , H ];
        end

        % 拼接整体设计矩阵（含全局截距）
        Phi = ones(N, 1 + sum(cellfun(@(A) size(A,2), Phi_parts)));
        col = 2;
        for j = 1:M
            pj = size(Phi_parts{j},2);
            Phi(:, col:col+pj-1) = Phi_parts{j};
            col = col + pj;
        end

        % 岭回归（不惩罚截距）
        P = size(Phi,2);
        pen = ones(P,1); pen(1) = 0;
        A = Phi.'*Phi + opts.lambdaRidge*diag(pen);
        B = Phi.'*X;
        % 数值稳定（优先Cholesky）
        try
            L = chol(A,'lower');
            Beta = L'\(L\B);
        catch
            Beta = A\B;
        end

        model.type   = 'GAM_linear_spline';
        model.knots  = knots;  % 1×M cell，每个是 1×Kj
        model.M      = M;
        model.Beta   = Beta;   % P×D
        model.opts   = opts;
    end

    %% ---------- GAM 预像：预测 ----------
    function Xhat = GAMPreimagePredict(model, Yq)
        % Yq: B×M（已归一化）
        B = size(Yq,1); M = model.M;
        % 重建与训练一致的设计矩阵
        Phi_parts = cell(1,M);
        for j = 1:M
            yj = Yq(:,j);
            t  = model.knots{j};
            H  = max(0, yj - t);
            Phi_parts{j} = [ yj , H ];
        end
        Phi = ones(B, 1 + sum(cellfun(@(A) size(A,2), Phi_parts)));
        col = 2;
        for j = 1:M
            pj = size(Phi_parts{j},2);
            Phi(:, col:col+pj-1) = Phi_parts{j};
            col = col + pj;
        end
        Xhat = Phi * model.Beta; % B×D
    end

    %% ---------- 工具 ----------
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
            D = sqrt(GAMUtils.pdist2_sq(A,B));
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
end
