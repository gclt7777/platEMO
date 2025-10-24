classdef TA_GAM < ALGORITHM
% <2025> <multi/many> <real/integer/label/binary/permutation>
% TA_GAM  Two-Archive + VaEA 环境选择 + （可选）AE潜空间生成
% 预像映射：GAM（分段线性样条的加性模型，逐维拟合 X <- Y）
%
% 参数：
%  alpha   —— CA 占比（默认 0.5）
%  nSeed   —— DA 极值种子数（默认 = M）
%  opMode  —— 0: 决策空间GA；1: 目标空间AE潜空间生成 + GAM预像
%  kLatent —— AE潜空间维数（默认 = min(M, max(2, ceil(M/2)))）
%  Knn     —— （兼容参数，已不使用）
%  etaC    —— 潜空间 SBX 的分布指数（默认 20）
%  etaM    —— 潜空间多项式变异的分布指数（默认 20）
%  pm      —— 潜空间逐基因变异概率（默认 1/kLatent）

methods
    function main(Algorithm, Problem)
        M = Problem.M;
        kDefault = min(M, max(2, ceil(M/2)));
        [alpha, nSeed, opMode, kLatent, Knn, etaC, etaM, pm] = ... %#ok<ASGLU>
            Algorithm.ParameterSet(0.5, M, 1, kDefault, 5, 20, 20, 1/max(2,kDefault));

        % 初始化
        Population = Problem.Initialization();
        N  = Problem.N;
        nCA = ceil(alpha*N);
        nDA = N - nCA;

        % 初次归一化边界（目标）
        [~, zmin, zmax] = GAMUtils.NormalizeObjs(Population.objs, [], []);

        % 初始 CA/DA
        CA = UpdateCA_GAM([], Population, nCA, zmin, zmax);
        DA = UpdateDA_GAM([], Population, nDA, zmin, zmax, nSeed);

        while Algorithm.NotTerminated([CA, DA])
            Pool = [CA, DA];

            if opMode == 0
                % —— 决策空间：通用 GA 生成 ——
                MatingIdx = randi(length(Pool), 1, 2*N);
                ParentDec = Pool(MatingIdx).decs;
                OffDec    = OperatorGA(Problem, ParentDec);
                Offspring = Problem.Evaluation(OffDec);
            else
                % —— 目标空间：AE潜空间生成 + GAM 预像 —— 
                UnionNow = [CA, DA];
                [ObjsN, zminAE, zmaxAE] = GAMUtils.NormalizeObjs(UnionNow.objs, [], []);
                [mu, W] = GAMUtils.FitLinearAE(ObjsN, kLatent);

                % 组配父代（与 GA 同样 2N 个）
                MatingIdx = randi(length(Pool), 1, 2*N);
                P1 = Pool(MatingIdx(1:N));  P2 = Pool(MatingIdx(N+1:2*N));

                % 编码潜空间
                Z1 = GAMUtils.AE_Encode(P1.objs, zminAE, zmaxAE, mu, W);
                Z2 = GAMUtils.AE_Encode(P2.objs, zminAE, zmaxAE, mu, W);

                % SBX + 多项式变异
                Zc = GAMUtils.SBX_PM(Z1, Z2, etaC, etaM, pm);

                % 解码到归一化目标 & 裁剪
                YN = GAMUtils.AE_Decode(Zc, mu, W);
                YN = min(max(YN,0),1);

                % —— GAM 预像映射（一次训练，整批预测）——
                repoObjsN = ObjsN;           % N_repo × M  （已归一化）
                repoDecs  = UnionNow.decs;   % N_repo × D
                gopt.numKnots   = 5;         % 每个目标维度的内部结点数（线性样条）
                gopt.lambdaRidge= 1e-4;      % 岭系数（不惩罚截距）
                gopt.knotRule   = 'quantile';% 结点策略：'quantile' | 'uniform'
                modelGAM = GAMUtils.GAMPreimageFit(repoObjsN, repoDecs, gopt);

                decsNew = GAMUtils.GAMPreimagePredict(modelGAM, YN);
                decsNew = min(max(decsNew, Problem.lower), Problem.upper);
                Offspring = Problem.Evaluation(decsNew);
            end

            % 更新归一化边界（目标）
            Union = [CA, DA, Offspring];
            [~, zmin, zmax] = GAMUtils.NormalizeObjs(Union.objs, [], []);

            % VaEA 风格两档更新
            CA = UpdateCA_GAM(CA, Union, nCA, zmin, zmax);
            DA = UpdateDA_GAM(DA, Union, nDA, zmin, zmax, nSeed);
        end
    end
end
end
