classdef OS_DNS_Soft < ALGORITHM
% <multi> <real/integer> <large/none>
% OS_DNS_Soft  Target-space autoencoder search with latent variation
%
% 本版本按照用户提出的“重新从头开始”思路实现：
%   1) 在目标空间中分别训练线性与非线性的自动编码器；
%   2) 在两个低维潜空间中对好解执行差分/突变，对坏解执行“向好解学习”；
%   3) 将线性/非线性自动编码器解码得到的新目标解混合；
%   4) 通过全局线性映射 + 局部邻域回归把目标解映射回决策空间，并保留精英用于
%      环境选择。
%
% 线性自动编码器使用 PCA 主成分作为编码/解码矩阵，非线性自动编码器使用带有
% tanh 激活的一层隐藏层，通过简单的批梯度下降训练。为了避免训练数据不足引起
% 的数值问题，当好解数量很少时自动退化为线性模型。映射阶段提供岭回归学习的
% 全局线性映射与 k 近邻的局部映射，并在两者之间进行线性插值。

    methods
        function main(Algorithm,Problem)
            %% 参数配置
            [latentRatio,latentMinDim,mixWeight,goodStep,badStep, ...
             noiseScale,mutationScale,eliteKeep,mapLambda,mapUpdateFreq, ...
             reuseMapping,kLocal,localMix,localSigma] = ...
                Algorithm.ParameterSet(0.5,2,0.5,0.6,0.35, ...
                                        0.01,0.1,0.1,1e-6,5, ...
                                        1,6,0.4,0.5);

            reuseMapping = reuseMapping ~= 0;
            latentRatio   = max(0.05,min(latentRatio,1));
            mixWeight     = max(0,min(mixWeight,1));
            goodStep      = max(0,goodStep);
            badStep       = max(0,badStep);
            noiseScale    = max(0,noiseScale);
            mutationScale = max(0,mutationScale);
            eliteKeep     = max(0,min(eliteKeep,0.5));
            kLocal        = max(0,round(kLocal));
            localMix      = max(0,min(localMix,1));
            localSigma    = max(1e-8,localSigma);

            %% 初始化种群
            Population = Problem.Initialization();
            N  = Problem.N;
            lb = Problem.lower;
            ub = Problem.upper;

            %% 映射器缓存
            T      = [];
            Ymean  = [];
            Ystd   = [];
            mapGen = 0;

            %% 主循环
            while Algorithm.NotTerminated(Population)
                Y = Population.objs;
                X = Population.decs;

                %% Step 1: 好解/坏解划分
                [goodMask,~] = OS_DNS_Soft.classify_population(Y,N);
                if ~any(goodMask)
                    % 至少保留一个好解
                    [~,bestIdx] = min(sum(Y.^2,2));
                    goodMask(bestIdx) = true;
                end
                badMask = ~goodMask;

                %% Step 2: 训练线性与非线性自动编码器
                goodY = Y(goodMask,:);
                if size(goodY,1) < 2
                    goodY = Y;
                end
                latentDim = max(latentMinDim,round(latentRatio*size(Y,2)));
                latentDim = min(latentDim,size(goodY,2));
                [linAE,linReady] = OS_DNS_Soft.train_linear_autoencoder(goodY,latentDim);
                [nonlinAE,nonlinReady] = OS_DNS_Soft.train_nonlinear_autoencoder(goodY,latentDim);
                if ~nonlinReady
                    % 回退到线性模型
                    nonlinAE = OS_DNS_Soft.wrap_linear_as_nonlinear(linAE);
                end

                %% Step 3: 在潜空间生成目标空间新解
                Ynew = OS_DNS_Soft.generate_target_offspring(Y,goodMask,linAE,nonlinAE, ...
                    mixWeight,goodStep,badStep,mutationScale,noiseScale,eliteKeep);

                %% Step 4: 目标空间 -> 决策空间映射
                if isempty(T) || (~reuseMapping) || (mapGen >= mapUpdateFreq)
                    [T,Ymean,Ystd] = OS_DNS_Soft.learn_linear_mapping(Y,X,mapLambda);
                    mapGen = 0;
                end
                mapGen = mapGen + 1;

                Xlinear = OS_DNS_Soft.map_linear(Ynew,T,Ymean,Ystd,lb,ub);
                if kLocal > 0 && localMix > 0 && size(Y,1) > 1
                    Xlocal = OS_DNS_Soft.map_local(Ynew,Y,X,kLocal,localSigma,lb,ub);
                    Xnew   = (1-localMix).*Xlinear + localMix.*Xlocal;
                else
                    Xnew   = Xlinear;
                end

                %% Step 5: 评估 + 环境选择
                Offspring = Problem.Evaluation(Xnew);
                Population = OS_DNS_Soft.env_select_nsga2([Population,Offspring],N);
            end
        end
    end

    methods (Static, Access = private)
        %% --- 人口分类 ---
        function [goodMask,badMask] = classify_population(Y,N)
            [FrontNo,MaxFNo] = NDSort(Y,N);
            goodMask = FrontNo == 1;
            if sum(goodMask) < max(2,ceil(0.2*size(Y,1)))
                goodMask = FrontNo <= min(MaxFNo,2);
            end
            badMask = ~goodMask;
        end

        %% --- 线性自动编码器 (PCA) ---
        function [model,ready] = train_linear_autoencoder(Y,latentDim)
            model = [];
            ready = false;
            if isempty(Y)
                return;
            end
            [n,m] = size(Y);
            latentDim = max(1,min(latentDim,min(n,m)));
            mu = mean(Y,1);
            Yc = Y - mu;
            C = (Yc' * Yc) / max(1,n-1);
            if latentDim >= m
                [Vall,Val] = eig((C+C')/2);
                [~,order] = sort(diag(Val),'descend');
                V = Vall(:,order(1:latentDim));
            else
                subspaceDim = min(m,max(latentDim+2,5));
                [V,~] = eigs((C+C')/2,latentDim,'largestreal','Tolerance',1e-6,'SubspaceDimension',subspaceDim);
            end
            model.type = 'linear';
            model.mean = mu;
            model.proj = V;
            ready = true;
        end

        %% --- 非线性自动编码器 (单隐层) ---
        function [model,ready] = train_nonlinear_autoencoder(Y,latentDim)
            model = [];
            ready = false;
            [n,m] = size(Y);
            if n < max(5,latentDim)
                return;
            end
            latentDim = max(1,min(latentDim,m));
            mu = mean(Y,1);
            Yc = Y - mu;
            % 初始化参数
            rngState = rng('shuffle'); %#ok<NASGU>
            Wenc = 0.1*randn(m,latentDim);
            benc = zeros(1,latentDim);
            Wdec = 0.1*randn(latentDim,m);
            bdec = zeros(1,m);
            lr = 0.01;
            weightDecay = 1e-4;
            maxIter = min(200,20*n);
            batchSize = min(64,n);
            for iter = 1:maxIter
                idx = randperm(n,batchSize);
                Ybatch = Yc(idx,:);
                Z = tanh(Ybatch*Wenc + benc);
                Yhat = Z*Wdec + bdec;
                Err = Yhat - Ybatch;
                gradWdec = (Z' * Err)/batchSize + weightDecay*Wdec;
                gradBdec = mean(Err,1);
                dZ = (Err * Wdec') .* (1 - Z.^2);
                gradWenc = (Ybatch' * dZ)/batchSize + weightDecay*Wenc;
                gradBenc = mean(dZ,1);

                Wdec = Wdec - lr*gradWdec;
                bdec = bdec - lr*gradBdec;
                Wenc = Wenc - lr*gradWenc;
                benc = benc - lr*gradBenc;

                if iter > 20 && mod(iter,20) == 0
                    lr = lr * 0.9;
                end
            end
            model.type = 'nonlinear';
            model.mean = mu;
            model.Wenc = Wenc;
            model.benc = benc;
            model.Wdec = Wdec;
            model.bdec = bdec;
            ready = true;
        end

        function model = wrap_linear_as_nonlinear(linModel)
            model.type = 'nonlinear';
            model.mean = linModel.mean;
            model.Wenc = linModel.proj;
            model.benc = zeros(1,size(linModel.proj,2));
            model.Wdec = linModel.proj';
            model.bdec = zeros(1,size(linModel.proj,1));
        end

        %% --- 编码/解码工具 ---
        function Z = encode_linear(Y,model)
            if isempty(model)
                Z = Y;
                return;
            end
            Yc = Y - model.mean;
            Z = Yc * model.proj;
        end

        function Yrec = decode_linear(Z,model)
            if isempty(model)
                Yrec = Z;
                return;
            end
            Yrec = Z * model.proj' + model.mean;
        end

        function Z = encode_nonlinear(Y,model)
            if isempty(model)
                Z = Y;
                return;
            end
            Yc = Y - model.mean;
            Z = tanh(Yc * model.Wenc + model.benc);
        end

        function Yrec = decode_nonlinear(Z,model)
            if isempty(model)
                Yrec = Z;
                return;
            end
            Yrec = Z * model.Wdec + model.bdec + model.mean;
        end

        %% --- 在潜空间生成新目标解 ---
        function Ynew = generate_target_offspring(Y,goodMask,linAE,nonlinAE,mixWeight, ...
                goodStep,badStep,mutationScale,noiseScale,eliteKeep)
            n = size(Y,1);
            M = size(Y,2);
            Ynew = Y;
            idxGood = find(goodMask);
            idxBad  = find(~goodMask);
            numGood = numel(idxGood);
            numBad  = numel(idxBad);

            if numGood == 0
                idxGood = (1:n)';
                numGood = n;
                idxBad  = [];
                numBad  = 0;
            end

            Yscale = max(std(Y,0,1),1e-6);

            %% 好解：差分 + 突变
            if numGood > 0
                Zg_lin = OS_DNS_Soft.encode_linear(Y(idxGood,:),linAE);
                Zg_non = OS_DNS_Soft.encode_nonlinear(Y(idxGood,:),nonlinAE);
                for t = 1:numGood
                    % 选取三个随机的好解用于差分组合
                    [a,b,c] = OS_DNS_Soft.sample_three(numGood,t);
                    zg_lin = Zg_lin(t,:) + goodStep*(Zg_lin(a,:) - Zg_lin(b,:));
                    zg_non = Zg_non(t,:) + goodStep*(Zg_non(a,:) - Zg_non(b,:));
                    if mutationScale > 0
                        zg_lin = zg_lin + mutationScale*randn(1,size(Zg_lin,2));
                        zg_non = zg_non + mutationScale*randn(1,size(Zg_non,2));
                    end
                    yg_lin = OS_DNS_Soft.decode_linear(zg_lin,linAE);
                    yg_non = OS_DNS_Soft.decode_nonlinear(zg_non,nonlinAE);
                    ymix   = mixWeight*yg_lin + (1-mixWeight)*yg_non;
                    if noiseScale > 0
                        ymix = ymix + noiseScale*randn(1,M).*Yscale;
                    end
                    Ynew(idxGood(t),:) = ymix;
                end
            end

            %% 坏解：向最近好解学习
            if numBad > 0 && numGood > 0
                Zg_lin = OS_DNS_Soft.encode_linear(Y(idxGood,:),linAE);
                Zb_lin = OS_DNS_Soft.encode_linear(Y(idxBad,:),linAE);
                Zg_non = OS_DNS_Soft.encode_nonlinear(Y(idxGood,:),nonlinAE);
                Zb_non = OS_DNS_Soft.encode_nonlinear(Y(idxBad,:),nonlinAE);
                nnIdx_lin = OS_DNS_Soft.nearest_indices(Zb_lin,Zg_lin);
                for t = 1:numBad
                    zg_lin = Zb_lin(t,:) + badStep*(Zg_lin(nnIdx_lin(t),:) - Zb_lin(t,:));
                    zg_non = Zb_non(t,:) + badStep*(Zg_non(nnIdx_lin(t),:) - Zb_non(t,:));
                    if mutationScale > 0
                        zg_lin = zg_lin + mutationScale*randn(1,size(Zb_lin,2));
                        zg_non = zg_non + mutationScale*randn(1,size(Zb_non,2));
                    end
                    yg_lin = OS_DNS_Soft.decode_linear(zg_lin,linAE);
                    yg_non = OS_DNS_Soft.decode_nonlinear(zg_non,nonlinAE);
                    ymix   = mixWeight*yg_lin + (1-mixWeight)*yg_non;
                    if noiseScale > 0
                        ymix = ymix + noiseScale*randn(1,M).*Yscale;
                    end
                    Ynew(idxBad(t),:) = ymix;
                end
            elseif numBad > 0
                % 没有好解时退化为扰动
                for t = 1:numBad
                    noise = noiseScale*randn(1,M).*Yscale;
                    Ynew(idxBad(t),:) = Y(idxBad(t),:) + noise;
                end
            end

            %% 精英保留：保留一部分原始的好解
            if eliteKeep > 0 && numGood > 0
                eliteNum = max(1,round(eliteKeep*numGood));
                eliteIdx = idxGood(1:min(eliteNum,numGood));
                Ynew(eliteIdx,:) = Y(eliteIdx,:);
            end
        end

        function [a,b,c] = sample_three(numCandidates,currentIdx)
            idx = 1:numCandidates;
            idx(idx==currentIdx) = [];
            if numel(idx) < 2
                a = currentIdx; b = currentIdx; c = currentIdx;
                return;
            end
            perm = idx(randperm(numel(idx)));
            a = perm(1);
            if numel(perm) >= 2
                b = perm(2);
            else
                b = perm(1);
            end
            if numel(perm) >= 3
                c = perm(3);
            else
                c = currentIdx;
            end
        end

        function nnIdx = nearest_indices(Zquery,Zref)
            nq = size(Zquery,1);
            nr = size(Zref,1);
            nnIdx = ones(nq,1);
            if nr == 0
                return;
            end
            for i = 1:nq
                diff = Zref - Zquery(i,:);
                dist = sum(diff.^2,2);
                [~,minIdx] = min(dist);
                nnIdx(i) = minIdx;
            end
        end

        %% --- 目标到决策空间映射 ---
        function [T,mu,sigma] = learn_linear_mapping(Y,X,lambda)
            [n,m] = size(Y);
            if nargin < 3
                lambda = 1e-6;
            end
            mu = mean(Y,1);
            sigma = std(Y,0,1);
            sigma(sigma < 1e-6) = 1;
            Yn = (Y - mu) ./ sigma;
            A = Yn' * Yn + lambda * eye(m);
            B = Yn' * X;
            T = A \ B;
        end

        function Xnew = map_linear(Y,T,mu,sigma,lb,ub)
            if isempty(T)
                Xnew = repmat((lb+ub)/2,size(Y,1),1);
                return;
            end
            Yn = (Y - mu) ./ sigma;
            Xnew = Yn * T;
            Xnew = OS_DNS_Soft.bound_correction(Xnew,lb,ub);
        end

        function Xlocal = map_local(Ynew,Y,X,k,sigmaScale,lb,ub)
            nNew = size(Ynew,1);
            Xlocal = zeros(nNew,size(X,2));
            for i = 1:nNew
                diff = Y - Ynew(i,:);
                dist = sum(diff.^2,2);
                [sorted,idx] = sort(dist);
                kUse = min(k,numel(idx));
                idx = idx(1:kUse);
                d = sorted(1:kUse);
                w = exp(-d/(sigmaScale*max(d(end),1e-6)));
                if all(w==0)
                    w(:) = 1;
                end
                w = w / sum(w);
                Xlocal(i,:) = sum(X(idx,:).*w,1);
            end
            Xlocal = OS_DNS_Soft.bound_correction(Xlocal,lb,ub);
        end

        function X = bound_correction(X,lb,ub)
            if ~isempty(lb)
                X = max(X,repmat(lb,size(X,1),1));
            end
            if ~isempty(ub)
                X = min(X,repmat(ub,size(X,1),1));
            end
        end

        %% --- 环境选择 (NSGA-II) ---
        function Population = env_select_nsga2(Population,N)
            if numel(Population) <= N
                Population = Population(1:min(N,numel(Population)));
                return;
            end
            Objs = Population.objs;
            [FrontNo,MaxFNo] = NDSort(Objs,N);
            Next = FrontNo < MaxFNo;
            Last = find(FrontNo==MaxFNo);
            CrowdDis = OS_DNS_Soft.crowding_distance(Objs(Last,:));
            [~,rank] = sort(CrowdDis,'descend');
            SelectedLast = Last(rank(1:(N-sum(Next))));
            Next(SelectedLast) = true;
            Population = Population(Next);
        end

        function CrowdDis = crowding_distance(Objs)
            [n,m] = size(Objs);
            CrowdDis = zeros(1,n);
            if n == 0
                return;
            end
            if n <= 2
                CrowdDis(:) = inf;
                return;
            end
            Fmax = max(Objs,[],1);
            Fmin = min(Objs,[],1);
            for i = 1:m
                [~,rank] = sort(Objs(:,i));
                CrowdDis(rank(1))   = inf;
                CrowdDis(rank(end)) = inf;
                for j = 2:n-1
                    prev = Objs(rank(j-1),i);
                    next = Objs(rank(j+1),i);
                    if Fmax(i) > Fmin(i)
                        CrowdDis(rank(j)) = CrowdDis(rank(j)) + (next - prev)/(Fmax(i)-Fmin(i));
                    else
                        CrowdDis(rank(j)) = CrowdDis(rank(j)) + 0;
                    end
                end
            end
        end
    end
end
