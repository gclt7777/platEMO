classdef TAELM < ALGORITHM
% <multi> <real/integer> <large/none>
% TAELM: Two-Archive + linear AE (PCA) + Local Linear Regression inverse mapping
% 生成：潜空间(线性AE)内 交叉 + 各向异性高斯变异 + TR 裁剪 → 解码回目标
% 逆映射：KNN-加权岭回归 (局部线性) 从目标 → 决策，再真实评估
% 环境选择：
%   CA：Iε⁺ 指标（wins, margin）主序，径向 r 次序，极端点必保
%   DA：先过滤与 CA 过近者，再用 L^{1/M} 的 max–min 贪心补满
%
% 依赖：AE_Update.m / AE_Encode.m / AE_Decode.m
%      LLR_InverseMap.m

    methods
        function main(Algorithm,Problem)
            %% 参数
            [NCA,NDA,N_off,latentDim,T_AE,lambdaLo,lambdaHi,beta,tau, ...
             kNN,ridge,KK,memCap] = Algorithm.ParameterSet( ...
                60, 60, [], ...
                max(3,min(10,Problem.M)), ...
                5, ...
                0.6, 1.0, ...
                0.10, 2.5, ...
                20, 1e-3, ...
                -1, 2000);

            if isempty(N_off) || N_off<=0, N_off = Problem.N; end
            if KK < 0, KK = max(12, round(1.2*NDA)); end
            if Problem.N ~= NCA + NDA, Problem.N = NCA + NDA; end

            %% 初始化
            Population   = Problem.Initialization();
            MemY         = Population.objs;
            MemX         = Population.decs;
            maximizeMask = false(1, Problem.M);   % 若有最大化目标，对应维设 true
            normState    = TAELM.InitYNormalizer(MemY, maximizeMask);

            [CA,DA] = TAELM.TwoArchive_Select(Population, NCA, NDA);
            AE = AE_Update([CA.objs; DA.objs], normState, struct('latentDim',latentDim));

            %% 主循环（本地代计数器 GEN）
            GEN = 1;
            while Algorithm.NotTerminated([CA,DA])
                % AE 更新（每 T_AE 代）
                if mod(GEN-1, T_AE) == 0
                    AE = AE_Update([CA.objs; DA.objs], normState, struct('latentDim',latentDim));
                end

                % 编码
                Z_CA = AE_Encode(CA.objs, AE);
                Z_DA = AE_Encode(DA.objs, AE);

                % 父代扇区（用于配对）
                W    = TAELM.GenRefDirs(Problem.M, KK);
                dirC = TAELM.AssignSector(TAELM.RobustScale(CA.objs), W);
                dirD = TAELM.AssignSector(TAELM.RobustScale(DA.objs), W);

                % 生成子代：潜空间交叉->变异->TR裁剪->解码
                Off_yhat = zeros(N_off, Problem.M);
                for t = 1:N_off
                    if isempty(DA)
                        idxD = randi(max(1,length(CA)));
                        kD   = randi(size(W,1));
                    else
                        kD   = dirD(randi(length(dirD)));
                        candD = find(dirD==kD); if isempty(candD), candD = 1:length(DA); end
                        idxD = candD(randi(numel(candD)));
                    end
                    nearS = unique([kD, mod(kD, size(W,1))+1, mod(kD-2, size(W,1))+1]);
                    candC = find(ismember(dirC, nearS)); if isempty(candC), candC = 1:length(CA); end
                    idxC  = candC(randi(numel(candC)));

                    zd = Z_DA(min(idxD,size(Z_DA,1)),:);
                    zc = Z_CA(min(idxC,size(Z_CA,1)),:);

                    lam = lambdaLo + (lambdaHi - lambdaLo)*rand();
                    z   = (1-lam)*zd + lam*zc;

                    z   = TAELM.MutateAniso(z, Z_CA, beta);
                    z   = TAELM.TrustRegionClip(z, Z_CA, tau);

                    Off_yhat(t,:) = AE_Decode(z, AE);
                end

                % 逆映射 -> 决策
                Off_xhat = zeros(N_off, Problem.D);
                for t = 1:N_off
                    Off_xhat(t,:) = LLR_InverseMap(Off_yhat(t,:), MemY, MemX, normState, ...
                                                   kNN, ridge, Problem.lower, Problem.upper, false);
                end

                % 真实评估
                Offspring = Problem.Evaluation(Off_xhat);

                % 环境选择（更新 CA/DA）
                U = [CA, DA, Offspring];
                [CA,DA] = TAELM.TwoArchive_Select(U, NCA, NDA);

                % 记忆库维护
                MemY = [MemY; Offspring.objs];  MemX = [MemX; Offspring.decs];
                if size(MemY,1) > memCap
                    cut = size(MemY,1) - memCap;
                    MemY(1:cut,:) = []; MemX(1:cut,:) = [];
                end

                GEN = GEN + 1;
            end
        end
    end

    %% 工具（静态）
    methods (Static, Access = private)
        % —— 统一最小化方向 + z-score 参数
        function normState = InitYNormalizer(Y, maximizeMask)
            signFlip = ones(1, size(Y,2));
            signFlip(logical(maximizeMask)) = -1;
            Ymin = Y .* signFlip;
            mu = mean(Ymin, 1);
            sigma = std(Ymin, 0, 1); sigma(sigma==0) = 1;
            normState.signFlip = signFlip;
            normState.mu = mu;
            normState.sigma = sigma;
        end

        % —— 环境选择：CA=Iε⁺，DA=过滤+L^{1/M} max–min
        function [CA,DA] = TwoArchive_Select(Pop, NCA, NDA)
            N  = length(Pop);
            M  = size(Pop(1).objs,2);
            Y  = Pop.objs;
            Yt = TAELM.RobustScale(Y);
            r  = sqrt(sum(Yt.^2,2));

            % CA：极端点 + Iε⁺ 排序
            ext    = TAELM.ExtremePoints(Y);
            ca_idx = unique(ext(:))';
            [wins, margin] = TAELM.EpsPlusStats(Yt);
            if numel(ca_idx) > NCA
                order  = TAELM.SortByEpsThenR(ca_idx, wins, margin, r);
                ca_idx = order(1:NCA);
            else
                order = TAELM.SortByEpsThenR(1:N, wins, margin, r);
                for id = order
                    if numel(ca_idx) >= NCA, break; end
                    if ~ismember(id, ca_idx)
                        ca_idx(end+1) = id; %#ok<AGROW>
                    end
                end
            end

            % DA：过滤与 CA 过近者 + L^{1/M} max–min
            rem_idx = setdiff(1:N, ca_idx);
            if isempty(rem_idx)
                da_idx = [];
            else
                p    = max(1e-6, 1/M);
                Dca  = TAELM.LpDist(Yt(rem_idx,:), Yt(ca_idx,:), p);
                d2CA = min(Dca, [], 2);
                thr  = quantile(d2CA, 0.20);
                cand = rem_idx(d2CA > thr);
                if isempty(cand)
                    [~,ordFar] = sort(d2CA, 'descend');
                    keep = min(max(NDA*3, NDA), numel(rem_idx));
                    cand = rem_idx(ordFar(1:keep));
                end
                da_idx = TAELM.GreedyMaxMin(Yt, ca_idx, cand, NDA, p);
            end

            if numel(da_idx) < NDA
                rest = setdiff(1:N, [ca_idx, da_idx]);
                need = min(NDA - numel(da_idx), numel(rest));
                if need > 0, da_idx = [da_idx, rest(1:need)]; end
            end

            CA = Pop(ca_idx);
            DA = Pop(da_idx);
        end

        % —— Iε⁺ 统计
        function [wins, margin] = EpsPlusStats(Yt)
            N = size(Yt,1);
            wins   = zeros(N,1);
            margin = zeros(N,1);
            for i = 1:N
                ei = max(bsxfun(@minus, Yt(i,:), Yt), [], 2);   % ε_ij
                ej = max(bsxfun(@minus, Yt, Yt(i,:)), [], 2);   % ε_ji
                better = ei < ej;
                wins(i)   = sum(better);
                margin(i) = sum(max(ej - ei, 0));
            end
        end

        % —— Iε⁺ 排序：wins↓, margin↓, r↑
        function order = SortByEpsThenR(idx, wins, margin, r)
            M = [-wins(idx), -margin(idx), r(idx)];
            [~,ord] = sortrows(M, [1 2 3]);
            order = idx(ord);
        end

        % —— L^p 距离（p=1/M）
        function D = LpDist(A, B, p)
            if isempty(A) || isempty(B)
                D = zeros(size(A,1), size(B,1));
                return;
            end
            n = size(A,1); k = size(B,1);
            D = zeros(n,k);
            for j = 1:k
                diff = abs(bsxfun(@minus, A, B(j,:))).^p;
                D(:,j) = sum(diff,2).^(1/p);
            end
        end

        % —— 基于 L^p 的 max–min 贪心（DA）
        function sel = GreedyMaxMin(Yt, ca_idx, cand_idx, K, p)
            if isempty(cand_idx)
                sel = cand_idx; return;
            end
            Dca   = TAELM.LpDist(Yt(cand_idx,:), Yt(ca_idx,:), p);
            d2CA  = min(Dca, [], 2);
            [~,i0] = max(d2CA);
            sel = cand_idx(i0);
            pool = setdiff(cand_idx, sel);

            while numel(sel) < K && ~isempty(pool)
                D    = TAELM.LpDist(Yt(pool,:), Yt(sel,:), p);
                dmin = min(D, [], 2);
                [~,ix] = max(dmin);
                sel(end+1) = pool(ix); %#ok<AGROW>
                pool(ix)   = [];
            end

            if numel(sel) < K
                rest = setdiff(cand_idx, sel);
                need = min(K - numel(sel), numel(rest));
                if need > 0, sel = [sel, rest(1:need)]; end
            end
        end

        % —— Robust [0,1] 缩放
        function Yt = RobustScale(Y)
            ymin = quantile(Y,0.01,1); ynad = quantile(Y,0.99,1);
            Yt = (Y - ymin)./(ynad - ymin + 1e-12);
        end

        % —— 极端点
        function idx = ExtremePoints(Y)
            [~, idx] = min(Y, [], 1);
            idx = unique(idx(:));
        end

        % —— 参考向量（父代配对用）
        function W = GenRefDirs(M, K)
            try
                [W,~] = UniformPoint(K, M);
            catch
                H = max(1, round(K^(1/(M-1))));
                W = TAELM.SimplexLattice(M,H);
            end
            W = W ./ max(vecnorm(W,2,2),1e-12);
        end

        function W = SimplexLattice(m,H)
            if m==1, W=1; return; end
            W=[];
            for i=0:H
                sub = TAELM.SimplexLattice(m-1, H-i);
                W=[W; [i/H*ones(size(sub,1),1), sub]];
            end
            W = W(sum(W,2)>0,:);  W = W ./ sum(W,2);
        end

        function dir = AssignSector(Yt, W)
            Yn = Yt ./ max(vecnorm(Yt,2,2), 1e-12);
            cosang = Yn * W';
            [~, dir] = max(cosang, [], 2);
        end

        % —— 各向异性高斯变异
        function z = MutateAniso(z, Zca, beta)
            d = size(Zca,2);
            if size(Zca,1)>=d+2
                C = cov(Zca); C = C + 1e-8*eye(d);
                [R,p] = chol(C,'lower');
                if p>0, z = z + beta*randn(1,d); return; end
                z = z + (randn(1,d)*R)*beta;
            else
                z = z + beta*randn(1,d);
            end
        end

        % —— Trust-Region 裁剪（马氏半径）
        function z = TrustRegionClip(z, Zca, tau)
            if size(Zca,1)<3, return; end
            C = cov(Zca); C = C + 1e-8*eye(size(C,1));
            mu = mean(Zca,1);
            v  = (z - mu) / C;
            dM = sqrt(v * (z - mu)');
            if dM > tau
                z = mu + (z - mu) * (tau/dM);
            end
        end
    end
end
