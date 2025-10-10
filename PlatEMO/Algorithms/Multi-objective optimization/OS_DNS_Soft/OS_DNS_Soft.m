classdef OS_DNS_Soft < ALGORITHM
% <multi> <real/integer> <large/none>
% OS_DNS_Soft  Soft-teacher driven target-space search with adaptive relaxation
%
% 思路：
%   1) 在目标空间区分“好解/坏解”，依据参考向量和 ε-支配关系构造软教师；
%   2) 好解朝理想点或扇区代表做精修，坏解按软教师学习；
%   3) 线性+局部预像将更新后的目标映射回决策空间；
%   4) 评估并结合网格拥挤度、参考向量指导的环境选择；必要时对拥挤坏解做重启。
%
% 参数（Algorithm.ParameterSet）：
%   alpha         [0.40]  坏解随教师前进的基准步长
%   beta          [0.15]  理想点牵引权重
%   lambdaT       [1e-6]  线性预像岭惩罚
%   tUpdateFreq   [1]     线性预像更新频率（代）
%   reuseT        [false] 是否复用旧 T（false 为每次更新重学）
%   epsRate       [0.01]  ε-支配阈值（相对每维范围）
%   kNeighbor     [10]    软教师候选近邻数
%   sigmaScale    [1.0]   高斯邻近权尺度
%   tauCheby      [0.5]   扇区偏好温度（0~1）
%   useRef        [1]     是否启用参考向量引导
%   envMode       [1]     环境选择：1=HypE，2=RVEA(APD)，其他=NSGA-II
%   kLocalMap     [6]     局部映射的近邻数（0 表示只用线性预像）
%   localMix      [0.5]   线性/局部映射混合权重
%   diversityRate [0.2]   额外 GA 变异体的比例
%   refAdaptFreq  [10]    参考向量自适应频率（代）
%   restartRate   [0.15]  拥挤坏解的重启比例
%   gridDiv       [8]     目标空间网格划分精度

    methods
        function main(Algorithm,Problem)
            %% 参数
            [alpha,beta,lambdaT,tUpdateFreq,reuseT, ...
             epsRate,kNeighbor,sigmaScale,tauCheby,useRef,envMode, ...
             kLocalMap,localMix,diversityRate,refAdaptFreq,restartRate,gridDiv] = ...
             Algorithm.ParameterSet(0.40,0.15,1e-6,1,false, ...
                                    0.01,10,1.0,0.5,1,1, ...
                                    6,0.5,0.2,10,0.15,8);

            %% 初始化
            Population = Problem.Initialization();
            N  = Problem.N;
            D  = Problem.D; %#ok<NASGU>
            lb = Problem.lower;
            ub = Problem.upper;

            Y = Population.objs;
            X = Population.decs;

            T = [];
            Ymean = [];
            Ystd  = [];

            if useRef
                [RefBase,~] = UniformPoint(max(N,Problem.M),Problem.M);
                RefVec  = RefBase;
            else
                RefBase = [];
                RefVec  = [];
            end
            sectorID = ones(size(Y,1),1);
            gen = 1;

            %% 主循环
            while Algorithm.NotTerminated(Population)
                % 1) 区分好解/坏解
                [FrontNo,MaxFNo] = NDSort(Y,N);
                goodMask = FrontNo==1;
                if sum(goodMask) < max(2,ceil(0.2*N))
                    goodMask = FrontNo <= min(MaxFNo,2);
                end
                if ~any(goodMask)
                    [~,best] = min(sum(Y,2));
                    goodMask(best) = true;
                end

                % 2) 参考向量自适应 & 扇区划分
                if useRef
                    [RefVec,sectorID] = OS_DNS_Soft.update_reference_vectors(Y, goodMask, RefVec, RefBase, gen, refAdaptFreq);
                else
                    RefVec  = [];
                    sectorID = ones(size(Y,1),1);
                end

                % 3) 构造软教师
                zmin = min(Y,[],1);
                [Yteacher,teacherInfo] = OS_DNS_Soft.construct_teachers(Y, goodMask, sectorID, RefVec, zmin, ...
                    epsRate,kNeighbor,sigmaScale,tauCheby,gridDiv);

                % 4) 目标空间松弛
                Y_new = OS_DNS_Soft.relax_population(Y,Yteacher,zmin,goodMask,alpha,beta);

                % 5) 线性+局部映射
                if gen==1 || (~reuseT && mod(gen-1,tUpdateFreq)==0)
                    [T,Ymean,Ystd] = OS_DNS_Soft.learnT_linear(Y,X,lambdaT);
                end
                X_linear = OS_DNS_Soft.mapY2X_linear(Y_new,T,lb,ub,Ymean,Ystd);
                if kLocalMap > 0 && localMix > 0
                    X_local  = OS_DNS_Soft.mapY2X_local(Y_new,Y,X,kLocalMap,lb,ub);
                    mixRatio = min(max(localMix,0),1);
                    X_new    = (1-mixRatio).*X_linear + mixRatio.*X_local;
                else
                    X_new = X_linear;
                end
                X_new = X_new + 1e-12*randn(size(X_new));

                % 6) 拥挤坏解重启
                if restartRate > 0
                    X_new = OS_DNS_Soft.restart_bad_solutions(X_new,X,lb,ub,~goodMask,teacherInfo.gridCrowd,restartRate);
                end

                % 7) 多样化注入
                if diversityRate > 0 && size(X,1) > 1
                    nOff = max(1,round(diversityRate*size(X_new,1)));
                    nOff = min(nOff,floor(size(X,1)/2));
                    if nOff > 0
                        parentIdx = randi(size(X,1),1,2*nOff);
                        parents   = X(parentIdx,:);
                        mutants   = OperatorGAhalf(Problem,parents);
                        replaceN  = min(size(mutants,1),size(X_new,1));
                        if replaceN > 0
                            replaceIdx = randperm(size(X_new,1),replaceN);
                            X_new(replaceIdx,:) = mutants(1:replaceN,:);
                        end
                    end
                end

                % 8) 评估+环境选择
                Offspring = Problem.Evaluation(X_new);
                PopAll    = [Population,Offspring];

                switch envMode
                    case 1
                        Population = env_select_hype(PopAll,N);
                    case 2
                        theta = 1;
                        if isprop(Algorithm,'FE') && isprop(Problem,'maxFE') && Problem.maxFE > 0
                            theta = min(2,max(0.5,2*(Algorithm.FE/Problem.maxFE)));
                        end
                        Population = env_select_rvea(PopAll,N,theta);
                    otherwise
                        Population = OS_DNS_Soft.envSelect_NSGA2(PopAll,N);
                end

                Y = Population.objs;
                X = Population.decs;
                gen = gen + 1;
            end
        end
    end

    %% === 辅助函数 ===
    methods (Static, Access = private)
        function [RefVec,sectorID] = update_reference_vectors(Y, goodMask, RefVec, RefBase, gen, adaptFreq)
            if isempty(RefBase)
                RefVec = [];
                sectorID = ones(size(Y,1),1);
                return;
            end
            if isempty(RefVec)
                RefVec = RefBase;
            end
            if gen == 1 || mod(gen-1,max(1,round(adaptFreq))) == 0
                refSource = Y(goodMask,:);
                if size(refSource,1) < size(Y,2)
                    refSource = Y;
                end
                try
                    RefVec = ReferenceVectorAdaptation(refSource, RefBase);
                catch
                    RefVec = RefBase;
                end
            end
            norms = sqrt(sum(RefVec.^2,2));
            norms(norms<1e-12) = 1;
            RefNorm = RefVec ./ norms;
            Yn = Y - min(Y,[],1);
            rngY = max(max(Y,[],1)-min(Y,[],1),1e-12);
            Yn = Yn ./ rngY;
            YnNorm = sqrt(sum(Yn.^2,2));
            YnNorm(YnNorm<1e-12) = 1;
            YnDir = Yn ./ YnNorm;
            simVal = YnDir * RefNorm';
            [~,sectorID] = max(simVal,[],2);
            sectorID(~isfinite(sectorID)) = 1;
        end

        function [Yteacher,info] = construct_teachers(Y, goodMask, sectorID, RefVec, zmin, epsRate,kNeighbor,sigmaScale,tauCheby,gridDiv)
            N = size(Y,1);
            M = size(Y,2);
            goodMask = logical(goodMask(:));
            badMask  = ~goodMask;

            Ymin = min(Y,[],1);
            Ymax = max(Y,[],1);
            Yrange = max(Ymax - Ymin, 1e-12);
            Yn = (Y - Ymin) ./ Yrange;
            epsVec = epsRate .* Yrange;

            [gridCrowd,gridID] = OS_DNS_Soft.grid_crowding(Yn,gridDiv);

            goodIdx = find(goodMask);
            badIdx  = find(badMask);
            Yteacher = Y;

            if isempty(goodIdx)
                goodIdx = (1:N)';
            end
            RefNorm = [];
            if ~isempty(RefVec)
                RefNorm = RefVec ./ max(1e-12,sqrt(sum(RefVec.^2,2)));
            end
            sectorBest = zeros(size(RefVec,1),1);
            if ~isempty(RefNorm)
                for r = 1:size(RefNorm,1)
                    cand = goodIdx(sectorID(goodIdx)==r);
                    if isempty(cand)
                        cand = goodIdx;
                    end
                    if isempty(cand)
                        continue;
                    end
                    candNorm = Yn(cand,:);
                    ref = max(RefNorm(r,:),1e-6);
                    cheby = max(bsxfun(@rdivide,candNorm,ref),[],2);
                    score = cheby + tauCheby*sum(candNorm,2);
                    [~,bestIdx] = min(score);
                    sectorBest(r) = cand(bestIdx);
                end
            end
            tauGood = min(max(tauCheby,0),0.6);
            for i = 1:length(goodIdx)
                idx = goodIdx(i);
                target = zmin;
                if ~isempty(sectorBest)
                    sec = sectorID(idx);
                    if sec <= numel(sectorBest) && sectorBest(sec) ~= 0
                        target = Y(sectorBest(sec),:);
                    end
                end
                Yteacher(idx,:) = (1 - tauGood).*Y(idx,:) + tauGood.*target;
            end

            if ~isempty(badIdx)
                try
                    dist = pdist2(Yn(badIdx,:),Yn(goodIdx,:));
                catch
                    dist = sqrt(max(0,sum(Yn(badIdx,:).^2,2) + sum(Yn(goodIdx,:).^2,2)' - 2*(Yn(badIdx,:)*Yn(goodIdx,:)')));
                end
                if isempty(dist)
                    dist = zeros(length(badIdx),length(goodIdx));
                end
                sigBase = median(dist(dist>0));
                if ~isfinite(sigBase) || sigBase <= 0
                    sigBase = 1;
                end
                for ii = 1:length(badIdx)
                    idx = badIdx(ii);
                    yi = Y(idx,:);
                    domMask = all(bsxfun(@le,Y(goodIdx,:),yi + epsVec + 1e-9),2) & ...
                              any(bsxfun(@lt,Y(goodIdx,:),yi - epsVec - 1e-9),2);
                    cand = goodIdx(domMask);
                    if isempty(cand)
                        if ~isempty(RefNorm)
                            sec = sectorID(idx);
                            if sec <= numel(sectorBest) && sectorBest(sec) ~= 0
                                cand = sectorBest(sec);
                            end
                        end
                    end
                    if isempty(cand)
                        [~,ord] = sort(dist(ii,:),'ascend');
                        kn = min(kNeighbor,length(ord));
                        cand = goodIdx(ord(1:kn));
                    end
                    cand = cand(:);
                    if isempty(cand)
                        cand = goodIdx(randi(length(goodIdx)));
                    end
                    candY = Y(cand,:);
                    weights = ones(numel(cand),1);
                    if numel(cand) > 1
                        d = sqrt(sum((Yn(cand,:) - Yn(idx,:)).^2,2));
                        sigma = max(sigmaScale*sigBase,1e-9);
                        weights = exp(-d.^2./(2*sigma^2));
                        crowdW  = 1./max(1,gridCrowd(cand));
                        weights = weights .* crowdW;
                        if ~isempty(RefNorm)
                            sec = sectorID(idx);
                            ref = RefNorm(max(1,min(sec,size(RefNorm,1))),:);
                            cheby = max(bsxfun(@rdivide,Yn(cand,:),max(ref,1e-6)),[],2);
                            weights = weights .* exp(-tauCheby*cheby);
                        end
                    end
                    if all(weights <= 0)
                        weights = ones(size(weights));
                    end
                    weights = weights ./ sum(weights);
                    Yteacher(idx,:) = (weights' * candY);
                end
            end

            info.goodMask  = goodMask;
            info.badMask   = badMask;
            info.gridCrowd = gridCrowd;
            info.gridID    = gridID;
        end

        function Y_new = relax_population(Y,Yteacher,zmin,goodMask,alpha,beta)
            N = size(Y,1);
            M = size(Y,2);
            goodMask = logical(goodMask(:));

            alphaGood = 0.6 * alpha;
            alphaBad  = 1.2 * alpha;
            betaGood  = 0.5 * beta;
            betaBad   = 1.5 * beta;

            alphaVec = alphaBad * ones(N,1);
            alphaVec(goodMask) = alphaGood;
            betaVec  = betaBad * ones(N,1);
            betaVec(goodMask)  = betaGood;

            alphaMat = repmat(alphaVec,1,M);
            betaMat  = repmat(betaVec,1,M);
            zMat     = repmat(zmin,N,1);

            Y_new = (1 - alphaMat).*Y + alphaMat.*((1 - betaMat).*Yteacher + betaMat.*zMat);

            overMask = sum(Y_new < min(Y,[],1)-1e-8,2) > 0 | sum(Y_new > max(Y,[],1)+1e-8,2) > 0;
            Y_new(overMask,:) = (Y(overMask,:) + Yteacher(overMask,:))/2;
        end

        function [T,Ymean,Ystd] = learnT_linear(Y,X,lambdaT)
            Ymean = mean(Y,1);
            Ystd  = std(Y,0,1);
            Ystd(Ystd < 1e-12) = 1;
            Yz = (Y - Ymean) ./ Ystd;
            M = size(Y,2);
            T = (Yz' * Yz + lambdaT * eye(M)) \ (Yz' * X);
        end

        function Xnew = mapY2X_linear(Yin,T,lb,ub,Ymean,Ystd)
            Yz = (Yin - Ymean) ./ Ystd;
            Xnew = Yz * T;
            [N,D] = size(Xnew);
            if isscalar(lb), lb = repmat(lb,1,D); end
            if isscalar(ub), ub = repmat(ub,1,D); end
            if iscolumn(lb), lb = lb'; end
            if iscolumn(ub), ub = ub'; end
            if numel(lb) ~= D, lb = repmat(lb(1),1,D); end
            if numel(ub) ~= D, ub = repmat(ub(1),1,D); end
            Xnew = min(repmat(ub,N,1), max(repmat(lb,N,1), Xnew));
        end

        function Xnew = mapY2X_local(Yin,Yref,Xref,kLocal,lb,ub)
            if isempty(Yref) || isempty(Xref)
                D = max([size(Xref,2), numel(lb), numel(ub)]);
                if D == 0
                    D = size(Yin,2);
                end
                lbv = lb; ubv = ub;
                if isscalar(lbv), lbv = repmat(lbv,1,D); end
                if isscalar(ubv), ubv = repmat(ubv,1,D); end
                if iscolumn(lbv), lbv = lbv'; end
                if iscolumn(ubv), ubv = ubv'; end
                mid = (lbv + ubv)/2;
                Xnew = repmat(mid,size(Yin,1),1);
                return;
            end
            kLocal = max(1,min(kLocal,size(Yref,1)));
            try
                dist = pdist2(Yin,Yref);
            catch
                dist = sqrt(max(0,sum(Yin.^2,2) + sum(Yref.^2,2)' - 2*(Yin*Yref')));
            end
            [~,order] = sort(dist,2);
            idx = order(:,1:kLocal);
            weights = zeros(size(idx));
            for i = 1:size(idx,1)
                di = dist(i,idx(i,:));
                if all(di < 1e-12)
                    w = double(di < 1e-12);
                    if ~any(w)
                        w(:) = 1;
                    end
                else
                    scale = median(di) + 1e-12;
                    w = exp(-di./scale);
                end
                wsum = sum(w);
                if wsum <= 0
                    w = ones(1,kLocal) ./ kLocal;
                else
                    w = w ./ wsum;
                end
                weights(i,:) = w;
            end
            Xnew = zeros(size(Yin,1), size(Xref,2));
            for i = 1:size(Yin,1)
                neighX = Xref(idx(i,:),:);
                Xnew(i,:) = weights(i,:) * neighX;
            end
            [N,D] = size(Xnew);
            if isscalar(lb), lb = repmat(lb,1,D); end
            if isscalar(ub), ub = repmat(ub,1,D); end
            if iscolumn(lb), lb = lb'; end
            if iscolumn(ub), ub = ub'; end
            if numel(lb) ~= D, lb = repmat(lb(1),1,D); end
            if numel(ub) ~= D, ub = repmat(ub(1),1,D); end
            Xnew = min(repmat(ub,N,1), max(repmat(lb,N,1), Xnew));
        end

        function [crowd,cellID] = grid_crowding(Yn,gridDiv)
            if nargin < 2 || gridDiv < 2
                gridDiv = 8;
            end
            Yn = max(0,min(1,Yn));
            bins = min(gridDiv-1, floor(Yn .* gridDiv));
            coeff = gridDiv.^((0:size(Yn,2)-1));
            cellID = bins * coeff' + 1;
            maxID = max(cellID);
            if isempty(maxID) || maxID < 1
                crowd = ones(size(cellID));
                return;
            end
            counts = accumarray(cellID,1,[maxID,1]);
            crowd = counts(cellID);
        end

        function Xnew = restart_bad_solutions(Xnew,Xold,lb,ub,badMask,gridCrowd,restartRate)
            badIdx = find(badMask);
            if isempty(badIdx) || restartRate <= 0
                return;
            end
            if isempty(gridCrowd)
                gridCrowd = ones(size(Xold,1),1);
            end
            [~,order] = sort(gridCrowd(badIdx),'descend');
            nRestart = min(numel(badIdx), max(1, round(restartRate * numel(badIdx))));
            sel = badIdx(order(1:nRestart));

            goodIdx = find(~badMask);
            if isempty(goodIdx)
                goodIdx = 1:size(Xold,1);
            end
            lbv = lb; ubv = ub;
            if isscalar(lbv), lbv = repmat(lbv,1,size(Xnew,2)); end
            if isscalar(ubv), ubv = repmat(ubv,1,size(Xnew,2)); end
            if iscolumn(lbv), lbv = lbv'; end
            if iscolumn(ubv), ubv = ubv'; end
            span = max(ubv - lbv, 1e-9);

            for i = 1:length(sel)
                teacher = Xold(goodIdx(randi(length(goodIdx))),:);
                noise   = 0.1 * span .* randn(1,size(Xnew,2));
                candidate = teacher + noise;
                candidate = min(max(candidate,lbv),ubv);
                Xnew(sel(i),:) = candidate;
            end
        end

        function PopOut = envSelect_NSGA2(PopIn,N)
            try
                Objs = PopIn.objs;
                [FrontNo,MaxFNo] = NDSort(Objs,N);
                Next = FrontNo < MaxFNo;
                CrowdDis = CrowdingDistance(Objs,FrontNo);
                Last = find(FrontNo==MaxFNo);
                [~,rank] = sort(CrowdDis(Last),'descend');
                Next(Last(rank(1:N-sum(Next)))) = true;
                PopOut = PopIn(Next);
            catch
                Objs = PopIn.objs;
                Ymin = min(Objs,[],1);
                Ymax = max(Objs,[],1);
                rngY = max(Ymax - Ymin, 1e-12);
                Yn = (Objs - Ymin) ./ rngY;
                score = sum(Yn,2);
                [~,ord] = sort(score,'ascend');
                PopOut = PopIn(ord(1:N));
            end
        end
    end
end
