classdef TA_VaEA < ALGORITHM
% <2025> <multi/many> <real/integer/label/binary/permutation>
% TA_VaEA  Two-Archive + VaEA 选择 + （可选）AE潜空间生成器
% 参数：
%  alpha   —— CA 占比（默认 0.5）
%  nSeed   —— DA 极值种子数（默认 = M）
%  opMode  —— 0: 决策空间GA（默认）；1: 目标空间AE潜空间生成
%  kLatent —— AE潜空间维数（默认 = min(M, max(2, ceil(M/2)))）
%  Knn     —— 逆映射的K近邻数（默认 5）
%  etaC    —— 潜空间 SBX 的分布指数（默认 20）
%  etaM    —— （保留参数位）不用于高斯突变，留作扩展（默认 20）
%  pm      —— 潜空间逐基因变异概率（默认 1/kLatent）

methods
    function main(Algorithm, Problem)
        M = Problem.M;
        kDefault = min(M, max(2, ceil(M/2)));
        [alpha, nSeed, opMode, kLatent, Knn, etaC, etaM, pm] = ...
            Algorithm.ParameterSet(0.6, M, 1, kDefault, 5, 20, 20, 1/max(2,kDefault));

        alphaBase = alpha;
        maxFE = max(Problem.maxFE, 1);

        % 初始化
        Population = Problem.Initialization();
        N  = Problem.N;
        nCA = max(1, min(N-1, ceil(alpha*N)));
        if nCA >= N
            nCA = N-1;
        end
        if nCA < M
            nCA = min(max(M,1), N-1);
        end
        nDA = max(1, N - nCA);
        if nCA + nDA > N
            nCA = N - nDA;
        end

        % 首次归一化边界（全局）
        [~, zmin, zmax] = VaEAUtils.NormalizeObjs(Population.objs, [], []);

        % 构建 CA / DA
        CA = UpdateCA_VaEA([], Population, nCA, zmin, zmax);
        DA = UpdateDA_VaEA([], Population, nDA, zmin, zmax, nSeed);

        while Algorithm.NotTerminated([CA, DA])
            Pool = [CA, DA];

            progress = min(max(Problem.FE/maxFE, 0), 1);

            if opMode == 0
                % —— 决策空间：沿用 PlatEMO 的 OperatorGA ——
                MatingIdx = randi(length(Pool), 1, 2*N);
                Offspring = OperatorGA(Pool(MatingIdx));
            else
                % —— 目标空间 AE 潜空间生成 + KNN 重心逆映射 ——
                % 基于当前 Union 的“归一化目标”拟合线性AE
                UnionNow = [CA, DA];
                [ObjsN, zminAE, zmaxAE] = VaEAUtils.NormalizeObjs(UnionNow.objs, [], []);
                [mu, W] = VaEAUtils.FitLinearAE(ObjsN, kLatent);

                % 组配父代（与 OperatorGA 同样取 2N 个）
                MatingIdx = randi(length(Pool), 1, 2*N);
                P1 = Pool(MatingIdx(1:N));
                P2 = Pool(MatingIdx(N+1:2*N));

                % 编码到潜空间 Z
                Z1 = VaEAUtils.AE_Encode(P1.objs, zminAE, zmaxAE, mu, W);
                Z2 = VaEAUtils.AE_Encode(P2.objs, zminAE, zmaxAE, mu, W);

                % 潜空间 SBX + 高斯突变（自适应方差）
                pmStage = max(0.05, pm*(1 - 0.5*progress));
                shrink  = max(0.2, 1 - 0.7*progress);
                Zc = VaEAUtils.SBX_PM(Z1, Z2, etaC, etaM, pmStage, shrink);

                % 解码回“归一化目标空间”并裁剪到[0,1]
                YN = VaEAUtils.AE_Decode(Zc, mu, W);

                % 用“全局 KNN + 重心”把目标估计映射回决策空间（边界修复）
                repoObjsN = ObjsN;                   % 全局库：当前种群的归一化目标
                repoDecs  = UnionNow.decs;           % 对应的决策
                decsNew   = VaEAUtils.KNNBaryMap(repoObjsN, repoDecs, YN, Knn, Problem.lower, Problem.upper);

                % 真实评估得到子代（以真实目标进入后续选择）
                Offspring = Problem.Evaluation(decsNew);
            end

            % 更新全局归一化边界
            Union = [CA, DA, Offspring];
            [~, zmin, zmax] = VaEAUtils.NormalizeObjs(Union.objs, [], []);

            % 两档更新（VaEA思想）
            progress = min(max(Problem.FE/maxFE, 0), 1);
            alphaNow = min(0.95, alphaBase + (1 - alphaBase)*progress);
            nCA = max(1, min(N-1, ceil(alphaNow*N)));
            if nCA >= N
                nCA = N-1;
            end
            if nCA < M
                nCA = min(max(M,1), N-1);
            end
            nDA = max(1, N - nCA);
            if nCA + nDA > N
                nCA = N - nDA;
            end

            CA = UpdateCA_VaEA(CA, Union, nCA, zmin, zmax);
            DA = UpdateDA_VaEA(DA, Union, nDA, zmin, zmax, nSeed);
            CA = VaEAUtils.RefineWithSeeds(CA, DA, zmin, zmax);
        end
    end
end
end
