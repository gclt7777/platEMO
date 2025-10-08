classdef AEKernelMapping < ALGORITHM
%<2025> <multi> <real>
% AEKERNELMAPPING  基于目标空间 DAE+Walkback 生成 + 核预像映射 的多目标算法
%   使用方式（PlatEMO 命令行示例）：
%     platemo('algorithm',@AEKernelMapping,'problem',@DTLZ2,'N',100,'maxFE',2e4);
%
% 核心流程：
%   1) 在目标空间 (Y) 训练广义 DAE + Walkback： learn P(Y | ~Y)
%   2) 用 “腐蚀 <-> 去噪” 链采样新目标向量 Y_new
%   3) 通过核岭回归的预像映射，将 Y_new 映回决策空间 X_new（带边界/约束）
%   4) 合并选择（内置 NSGA-II 环境选择，含约束支配）
%
% 依赖：
%   - 本目录下：trainDAE_walkback.m, sampleY_walkback.m, energyFilterY.m, preimage_kernel_map.m
%   - Deep Learning Toolbox（dlnetwork）与 Optimization Toolbox（若使用 fmincon；否则用内置 PGD 备选）
%
% 作者提示：为了易用，类内部自带一个简化版 NSGA-II 选择（NDSort+拥挤度）。
%           若你有自定义选择器，可替换 selectNSGA2_ 内部实现。

properties (SetAccess = private)
    % DAE + Walkback 模型与配置
    WBModel      % 由 trainDAE_walkback 返回的结构体
    WBOpts       % 训练配置（见 defaultWBOpts_）

    % 预像映射配置（核岭 + box 约束优化）
    KernelParams % 见 defaultKernelParams_

    % 其它
    UpdateEvery = 10;  % 每隔多少代更新一次 DAE（首代必训）
    FilterQuantile = 0.7; % 采样后能量/重构过滤保留比例
    SampleCondSigma = 0.0; % decode 后额外小噪声
end

methods
    function main(Algorithm, Problem)
        % ===== 0) 初始化 =====
        rng('shuffle');
        N = Problem.N;
        D = Problem.D; %#ok<NASGU>
        M = Problem.M;

        % 初始化种群（若 Problem 有 Initialization 则用；否则随机）
        try
            Population = Problem.Initialization();
        catch
            X0 = rand(N, Problem.D) .* (Problem.upper - Problem.lower) + Problem.lower;
            PopObj = Problem.CalObj(X0);
            if isprop(Problem,'CalCon')
                PopCon = Problem.CalCon(X0);
            else
                PopCon = zeros(N,1);
            end
            Population = SOLUTION(X0, PopObj, PopCon);
        end

        % 默认参数
        if isempty(Algorithm.WBOpts)
            Algorithm.WBOpts = Algorithm.defaultWBOpts_(M);
        end
        if isempty(Algorithm.KernelParams)
            Algorithm.KernelParams = Algorithm.defaultKernelParams_(Problem);
        end

        gen = 0;
        % ===== 1) 迭代 =====
        while Algorithm.NotTerminated(Population)
            gen = gen + 1;

            % —— 1.1 训练/更新 DAE+Walkback ——
            if gen == 1 || mod(gen, Algorithm.UpdateEvery) == 0 || isempty(Algorithm.WBModel)
                Y = Population.objs;
                Algorithm.WBModel = trainDAE_walkback(Y, Algorithm.WBOpts);
            end

            % —— 1.2 在目标空间采样新 Y ——
            Yseed = Population.objs(randperm(length(Population), Problem.N), :);
            Ynew  = sampleY_walkback(Algorithm.WBModel, Yseed, struct('stepsMin',5,'stepsMax',20,'condSigma',Algorithm.SampleCondSigma));

            % —— 1.3 可选过滤（能量/重构阈值） ——
            if Algorithm.FilterQuantile > 0 && Algorithm.FilterQuantile < 1
                keep = energyFilterY(Algorithm.WBModel, Ynew, struct('keepQuantile',Algorithm.FilterQuantile));
                Ynew = Ynew(keep,:);
                if isempty(Ynew) % 保底
                    Ynew = Yseed;
                end
            end

            % —— 1.4 预像映射：Y_new -> X_new ——
            Xref = Population.decs;
            Yref = Population.objs;
            Bounds.lb = Problem.lower; Bounds.ub = Problem.upper;
            Xnew = preimage_kernel_map(Ynew, Xref, Yref, Algorithm.KernelParams, Bounds);

            % —— 1.5 评估 & 构建子代 ——
            OObj = Problem.CalObj(Xnew);
            if isprop(Problem,'CalCon')
                OCon = Problem.CalCon(Xnew);
            else
                OCon = zeros(size(Xnew,1),1);
            end
            Offspring = SOLUTION(Xnew, OObj, OCon);

            % —— 1.6 合并选择（简化 NSGA-II） ——
            Population = Algorithm.selectNSGA2_([Population, Offspring], Problem.N);
        end
    end
end

methods (Access = private)
    function opts = defaultWBOpts_(Algorithm, M) %#ok<INUSD>
        opts = struct();
        opts.latentDim       = max(6, round(1.2*M));
        opts.noiseSigmaRatio = [0.05 0.15];
        opts.maskProb        = 0.0;
        opts.walk_p          = 0.5;
        opts.walk_trajs      = 1;
        opts.epochs          = 60;
        opts.batchSize       = 128;
        opts.learnRate       = 1e-3;
        opts.weightDecay     = 0; % 可设 1e-6
        opts.gpu             = canUseGPU;
        opts.verbose         = false;
    end

    function params = defaultKernelParams_(Algorithm, Problem) %#ok<INUSD>
        params = struct();
        params.kernel   = 'rbf';
        params.sigma    = 0.2 * mean(Problem.upper - Problem.lower); % RBF 宽度
        params.lambda   = 1e-3;   % KRR 正则
        params.beta     = 1e-3;   % 预像正则（靠近参考点，平滑）
        params.maxIter  = 200;
        params.restarts = 1;
        params.useFmincon = license('test','Optimization_Toolbox');
        params.verbose  = false;
    end

    function Pop = selectNSGA2_(Algorithm, Pop, N) %#ok<INUSD>
        % 简化版 NSGA-II 环境选择（带约束支配）
        % 输入 Pop 是 SOLUTION 数组
        PopObj = cat(1, Pop.objs);
        if any(arrayfun(@(s) ~isempty(s.cons), Pop))
            Cons = cat(1, Pop.cons);
            CV = sum(max(0,Cons),2);
        else
            CV = zeros(size(PopObj,1),1);
        end

        % 非支配排序（约束优先）
        [FrontNo, MaxFNo] = ndSortCon_(PopObj, CV, N);
        Next = FrontNo < MaxFNo;
        Pop_next = Pop(Next);
        if sum(Next) < N
            Last = find(FrontNo == MaxFNo);
            CD = crowdingDistance_(PopObj(Last,:), CV(Last));
            [~, rank] = sort(CD, 'descend');
            Pop_next = [Pop_next, Pop(Last(rank(1:N - sum(Next))))];
        end
        Pop = Pop_next;
    end
end

methods (Static, Access = private)
    function [FrontNo, MaxFNo] = ndSortCon_(Obj, CV, N)
        % 约束优先的非支配排序
        M = size(Obj,2);
        PopSize = size(Obj,1);
        FrontNo = inf(PopSize,1);
        MaxFNo  = 0;

        Fea = find(CV<=1e-12);
        Infea = find(CV>1e-12);
        % 可行解先进行标准 NDSort
        if ~isempty(Fea)
            [FrontNoFea, MaxFea] = NDSort(Obj(Fea,:), N);
            FrontNo(Fea) = FrontNoFea;
            MaxFNo = max(MaxFNo, MaxFea);
        end
        % 不可行解按 CV 排序，接在后面
        if ~isempty(Infea)
            [~, order] = sort(CV(Infea), 'ascend');
            ranks = (1:length(Infea))' + MaxFNo;
            FrontNo(Infea(order)) = ranks;
            MaxFNo = ranks(end);
        end
        % 裁剪最大前沿编号
        if sum(FrontNo<=N) < N
            MaxFNo = min(MaxFNo, max(FrontNo(FrontNo<=N)));
        end
    end

    function CD = crowdingDistance_(Obj, CV)
        % 拥挤度（对可行解按目标计算；对不可行解按 CV 扩展）
        PopSize = size(Obj,1);
        M = size(Obj,2);
        CD = zeros(PopSize,1);
        Fea = CV<=1e-12;
        if any(Fea)
            FeaIdx = find(Fea);
            ObjFea = Obj(Fea,:);
            Nf = size(ObjFea,1);
            if Nf>0
                fmax = max(ObjFea,[],1); fmin = min(ObjFea,[],1);
                span = max(fmax - fmin, 1e-12);
                for m = 1:M
                    [~,ord] = sort(ObjFea(:,m));
                    CD(FeaIdx(ord(1)))   = inf;
                    CD(FeaIdx(ord(end))) = inf;
                    for i = 2:Nf-1
                        CD(FeaIdx(ord(i))) = CD(FeaIdx(ord(i))) + (ObjFea(ord(i+1),m)-ObjFea(ord(i-1),m))/span(m);
                    end
                end
            end
        end
        if any(~Fea)
            idx = find(~Fea);
            % 不可行解按更小 CV 更优，CD 用 -CV 使其有序
            CD(idx) = -CV(idx);
        end
    end
end

end
