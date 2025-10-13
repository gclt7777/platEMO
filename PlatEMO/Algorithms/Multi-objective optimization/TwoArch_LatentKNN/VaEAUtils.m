classdef VaEAUtils
% 工具函数集合：归一化/fit/极值/角度贪心 + 线性AE + 潜空间遗传 + KNN重心逆映射
methods(Static)
    %% —— 归一化/评价 —— %%
    function [ObjsN, zmin, zmax] = NormalizeObjs(Objs, zmin, zmax)
        if nargin < 2 || isempty(zmin), zmin = min(Objs, [], 1); end
        if nargin < 3 || isempty(zmax), zmax = max(Objs, [], 1); end
        den   = max(zmax - zmin, eps);
        ObjsN = (Objs - zmin) ./ den;
        ObjsN = max(min(ObjsN, 1), 0); % 裁剪
    end

    function fit = FitScore(ObjsN)
        fit = sum(ObjsN, 2); % 越小越好
    end

    function seedIdx = ExtremeSeeds(ObjsN, K)
        [~, idxMin] = min(ObjsN, [], 1);
        seedIdx = unique(idxMin(:)', 'stable');
        if numel(seedIdx) < K
            fit = sum(ObjsN,2);
            [~, ord] = sort(fit, 'ascend');
            seedIdx  = unique([seedIdx, ord(:)'], 'stable');
        end
        seedIdx = seedIdx(1:min(K, numel(seedIdx)));
    end

    function sel = AngleGreedy(ObjsN, quota, initSel)
        N = size(ObjsN,1);
        norms = sqrt(sum(ObjsN.^2, 2)) + eps;
        U = ObjsN ./ norms;
        if nargin < 3 || isempty(initSel)
            fit = sum(ObjsN,2);
            [~, start] = min(fit);
            sel = start;
        else
            sel = unique(initSel(:)', 'stable');
        end
        cand = setdiff(1:N, sel);
        while numel(sel) < quota && ~isempty(cand)
            cosVals = U(cand,:)*U(sel,:)';
            cosVals = min(max(cosVals, 0), 1);
            ang     = real(acos(cosVals));
            minAng  = min(ang, [], 2);
            [~, idx] = max(minAng);
            sel = [sel, cand(idx)]; %#ok<AGROW>
            cand(idx) = [];
        end
        if numel(sel) > quota
            cosVals = U(sel,:)*U(sel,:)';
            cosVals = min(max(cosVals, 0), 1);
            angMat  = real(acos(cosVals));
            avgAng  = mean(angMat, 2);
            [~, ord] = sort(avgAng, 'descend');
            sel = sel(ord(1:quota));
        end
    end

    %% —— 线性 AE（PCA 等价） —— %%
    function [mu, W] = FitLinearAE(ObjsN, k)
        % 输入：ObjsN (N x M, 已归一化到[0,1])
        % 输出：mu (1 x M), W (M x k), 使得 encode: Z=(Y-mu)*W, decode: Y=Z*W'+mu
        mu = mean(ObjsN, 1);
        Yc = ObjsN - mu;                       % 中心化
        [~, ~, V] = svd(Yc, 'econ');           % PCA 主方向
        k = min(size(V,2), k);
        W = V(:, 1:k);
    end

    function Z = AE_Encode(Objs, zmin, zmax, mu, W)
        [Y, ~, ~] = VaEAUtils.NormalizeObjs(Objs, zmin, zmax);
        Yc = bsxfun(@minus, Y, mu);
        Z  = Yc * W;
    end

    function YN = AE_Decode(Z, mu, W)
        YN = Z * W' + mu;                      % 回到“归一化目标空间”
        YN = max(min(YN, 1), 0);               % 裁剪到[0,1]
    end

    %% —— 潜空间遗传：SBX + 高斯突变 —— %%
    function Zc = SBX_PM(Z1, Z2, etaC, ~, pm)
        % SBX 产生一个子代（逐行在 c1/c2 中任选）
        U = rand(size(Z1));
        betaQ = zeros(size(Z1));
        idx = U <= 0.5;
        betaQ(idx)  = (2.*U(idx)).^(1./(etaC+1));
        betaQ(~idx) = (1./(2.*(1-U(~idx)))).^(1./(etaC+1));
        C1 = 0.5*((1+betaQ).*Z1 + (1-betaQ).*Z2);
        C2 = 0.5*((1-betaQ).*Z1 + (1+betaQ).*Z2);
        pick = rand(size(Z1,1),1) < 0.5;
        Zc   = C1;
        Zc(pick,:) = C2(pick,:);

        % 自适应高斯突变（方差按父代联合标准差的 0.15 倍）
        Zstack = [Z1; Z2];
        sigma  = 0.15*std(Zstack, 0, 1);
        sigma  = max(sigma, 1e-12);
        noise  = randn(size(Zc)) .* sigma;
        mask   = rand(size(Zc)) < pm;
        Zc     = Zc + mask.*noise;
    end

    %% —— 目标→决策：全局 KNN + 重心映射（无核） —— %%
    function decsNew = KNNBaryMap(repoObjsN, repoDecs, YN, K, lower, upper)
        % repoObjsN: (R x M)  归一化目标库
        % repoDecs : (R x D)  对应决策库
        % YN       : (N x M)  待映射的目标（已归一化）
        % 返回 decsNew: (N x D)
        [R, D] = size(repoDecs);
        N = size(YN,1);
        K = min(K, R);
        decsNew = zeros(N, D);

        for i = 1:N
            y = YN(i,:);
            % 欧氏距离 KNN
            diff = repoObjsN - y;
            d2   = sum(diff.^2, 2);
            [~, ord] = sort(d2, 'ascend');
            nn  = ord(1:K);
            di  = sqrt(d2(nn));
            if all(di < 1e-12)
                w = zeros(K,1); w(1) = 1; % 完全匹配，取最近一个
            else
                w = 1./(di + 1e-12);
                w = w ./ sum(w);
            end
            xhat = w' * repoDecs(nn, :); % 重心
            decsNew(i,:) = xhat;
        end
        % 边界修复
        if ~isempty(lower) && ~isempty(upper)
            lower = lower(:)'; upper = upper(:)';
            if numel(lower) == 1, lower = repmat(lower, 1, D); end
            if numel(upper) == 1, upper = repmat(upper, 1, D); end
            decsNew = min(max(decsNew, lower), upper);
        end
    end
end
end
