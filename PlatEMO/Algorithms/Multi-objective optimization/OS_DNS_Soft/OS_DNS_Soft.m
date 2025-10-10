classdef OS_DNS_Soft < ALGORITHM
% <multi> <real/integer> <large/none>
% OS_DNS_Soft  Soft-teacher driven target-space search with adaptive control
%
% 该版本根据《OS_DNS_Soft 收敛性分析》与《OS_DNS_Soft 后续改进建议》重写，
% 在目标空间执行“好解/坏解”分层、参考向量引导的软教师学习、网格拥挤
% 控制与动态重启，再结合线性+局部映射回到决策空间并进行环境选择。

    methods
        function main(Algorithm,Problem)
            %% 参数
            [alpha,beta,lambdaT,updateFreq,reuseT, ...
             epsRate,kNeighbor,sigmaScale,tauCheby,useRef,envMode, ...
             kLocal,localMix,diversityRate,refAdaptFreq,restartRate,gridDiv] = ...
             Algorithm.ParameterSet(0.35,0.15,1e-6,1,false, ...
                                    0.02,10,1.0,0.5,1,1, ...
                                    8,0.5,0.15,10,0.15,8);

            %% 初始化
            Population = Problem.Initialization();
            N  = Problem.N;
            lb = Problem.lower;
            ub = Problem.upper;

            Y = Population.objs;
            X = Population.decs;

            T = [];
            Ymean = [];
            Ystd  = [];

            if useRef
                [RefBase,~] = UniformPoint(max(N,Problem.M),Problem.M);
                RefVec = RefBase;
            else
                RefBase = [];
                RefVec  = [];
            end
            sectorID = ones(size(Y,1),1);
            gen = 1;

            %% 主循环
            while Algorithm.NotTerminated(Population)
                % 1) 区分好解与坏解
                [goodMask,badMask] = OS_DNS_Soft.classify_population(Y,N);

                % 2) 自适应参考向量与扇区划分
                if useRef
                    [RefVec,sectorID] = OS_DNS_Soft.adapt_reference_vectors(Y, goodMask, RefVec, RefBase, gen, refAdaptFreq);
                else
                    RefVec  = [];
                    sectorID = ones(size(Y,1),1);
                end

                % 3) 计算网格拥挤度，构造软教师
                zmin = min(Y,[],1);
                gridInfo = OS_DNS_Soft.compute_grid_info(Y, gridDiv);
                Yteacher = OS_DNS_Soft.build_soft_teachers(Y, goodMask, badMask, sectorID, RefVec, ...
                    zmin, gridInfo, epsRate, kNeighbor, sigmaScale, tauCheby);

                % 4) 目标空间松弛（好解/坏解不同步长）
                Ynew = OS_DNS_Soft.relax_population(Y,Yteacher,zmin,goodMask,alpha,beta);

                % 5) 线性映射与局部映射混合
                if gen==1 || (~reuseT && mod(gen-1,updateFreq)==0)
                    [T,Ymean,Ystd] = OS_DNS_Soft.learn_linear_mapping(Y,X,lambdaT);
                elseif reuseT && isempty(T)
                    [T,Ymean,Ystd] = OS_DNS_Soft.learn_linear_mapping(Y,X,lambdaT);
                end
                Xlinear = OS_DNS_Soft.map_linear(Ynew,T,Ymean,Ystd,lb,ub);

                if kLocal > 0 && localMix > 0
                    Xlocal  = OS_DNS_Soft.map_local(Ynew,Y,X,kLocal,lb,ub);
                    mixRatio = min(max(localMix,0),1);
                    Xnew = (1-mixRatio).*Xlinear + mixRatio.*Xlocal;
                else
                    Xnew = Xlinear;
                end

                % 6) 拥挤坏解动态重启
                if restartRate > 0
                    Xnew = OS_DNS_Soft.restart_bad_solutions(Xnew,X,goodMask,gridInfo,restartRate,lb,ub);
                end

                % 7) 多样化注入（GA 变异）
                if diversityRate > 0 && size(X,1) > 1
                    Xnew = OS_DNS_Soft.inject_diversity(Xnew,X,diversityRate,Problem);
                end

                % 8) 评估 + 环境选择
                Offspring = Problem.Evaluation(Xnew);
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
                        Population = OS_DNS_Soft.env_select_nsga2(PopAll,N);
                end

                Y = Population.objs;
                X = Population.decs;
                gen = gen + 1;
            end
        end
    end

    %% ===== 静态辅助函数 =====
    methods (Static, Access = private)
        function [goodMask,badMask] = classify_population(Y,N)
            [FrontNo,MaxFNo] = NDSort(Y,N);
            goodMask = FrontNo==1;
            if sum(goodMask) < max(2,ceil(0.2*size(Y,1)))
                goodMask = FrontNo <= min(MaxFNo,2);
            end
            if ~any(goodMask)
                [~,best] = min(sum(Y,2));
                goodMask(best) = true;
            end
            badMask = ~goodMask;
        end

        function [RefVec,sectorID] = adapt_reference_vectors(Y,goodMask,RefVec,RefBase,gen,adaptFreq)
            if isempty(RefBase)
                RefVec = [];
                sectorID = ones(size(Y,1),1);
                return;
            end
            if isempty(RefVec)
                RefVec = RefBase;
            end
            freq = max(1,round(adaptFreq));
            if gen==1 || mod(gen-1,freq)==0
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

            Yshift = Y - min(Y,[],1);
            denom  = max(max(Y,[],1)-min(Y,[],1), 1e-12);
            Yn = Yshift ./ denom;
            YnNorm = sqrt(sum(Yn.^2,2));
            YnNorm(YnNorm<1e-12) = 1;
            YnDir = Yn ./ YnNorm;

            sim = YnDir * RefNorm';
            [~,sectorID] = max(sim,[],2);
            sectorID(~isfinite(sectorID)) = 1;
        end

        function gridInfo = compute_grid_info(Y,gridDiv)
            if nargin < 2 || gridDiv < 2
                gridDiv = 8;
            end
            Ymin = min(Y,[],1);
            Yrange = max(max(Y,[],1) - Ymin, 1e-12);
            Yn = (Y - Ymin) ./ Yrange;
            Yn = min(max(Yn,0),1);

            bins = min(gridDiv-1, floor(Yn .* gridDiv));
            coeff = gridDiv.^((0:size(Y,2)-1));
            cellID = bins * coeff' + 1;
            maxID = max(cellID);
            if isempty(maxID) || maxID < 1
                crowd = ones(size(cellID));
            else
                counts = accumarray(cellID,1,[maxID,1]);
                crowd = counts(cellID);
            end

            gridInfo.Yn      = Yn;
            gridInfo.cellID  = cellID;
            gridInfo.crowd   = crowd;
            gridInfo.Ymin    = Ymin;
            gridInfo.Yrange  = Yrange;
        end

        function Yteacher = build_soft_teachers(Y,goodMask,badMask,sectorID,RefVec,zmin,gridInfo,epsRate,kNeighbor,sigmaScale,tauCheby)
            Yn = gridInfo.Yn;
            epsVec = epsRate .* gridInfo.Yrange;

            goodIdx = find(goodMask);
            badIdx  = find(badMask);
            Yteacher = Y;

            if isempty(goodIdx)
                goodIdx = (1:size(Y,1))';
            end

            RefNorm = [];
            if ~isempty(RefVec)
                RefNorm = RefVec ./ max(1e-12,sqrt(sum(RefVec.^2,2)));
            end

            % 好解向理想点或扇区代表靠拢
            tauGood = min(max(tauCheby,0),0.6);
            sectorBest = OS_DNS_Soft.pick_sector_elites(Y,Yn,goodIdx,sectorID,RefNorm,tauCheby);
            for i = 1:numel(goodIdx)
                idx = goodIdx(i);
                target = zmin;
                if ~isempty(sectorBest)
                    sec = sectorID(idx);
                    if sec <= numel(sectorBest) && sectorBest(sec) > 0
                        target = Y(sectorBest(sec),:);
                    end
                end
                Yteacher(idx,:) = (1-tauGood)*Y(idx,:) + tauGood*target;
            end

            if isempty(badIdx)
                return;
            end

            try
                dist = pdist2(Yn(badIdx,:),Yn(goodIdx,:));
            catch
                dist = sqrt(max(0,sum(Yn(badIdx,:).^2,2) + sum(Yn(goodIdx,:).^2,2)' - 2*(Yn(badIdx,:)*Yn(goodIdx,:)')));
            end
            if isempty(dist)
                dist = zeros(numel(badIdx),numel(goodIdx));
            end
            sigBase = median(dist(dist>0));
            if ~isfinite(sigBase) || sigBase <= 0
                sigBase = 1;
            end

            for ii = 1:numel(badIdx)
                idx = badIdx(ii);
                yi  = Y(idx,:);

                domMask = all(bsxfun(@le,Y(goodIdx,:),yi + epsVec + 1e-9),2) & ...
                          any(bsxfun(@lt,Y(goodIdx,:),yi - epsVec - 1e-9),2);
                cand = goodIdx(domMask);

                if isempty(cand) && ~isempty(sectorBest)
                    sec = sectorID(idx);
                    if sec <= numel(sectorBest) && sectorBest(sec) > 0
                        cand = sectorBest(sec);
                    end
                end

                if isempty(cand)
                    [~,ord] = sort(dist(ii,:),'ascend');
                    kn = min(kNeighbor,length(ord));
                    cand = goodIdx(ord(1:kn));
                end

                cand = cand(:);
                if isempty(cand)
                    cand = goodIdx(randi(numel(goodIdx)));
                end

                candY = Y(cand,:);
                weights = ones(numel(cand),1);
                if numel(cand) > 1
                    d = sqrt(sum((Yn(cand,:) - Yn(idx,:)).^2,2));
                    sigma = max(sigmaScale*sigBase,1e-9);
                    weights = exp(-d.^2./(2*sigma^2));
                    crowdW  = 1 ./ max(1,gridInfo.crowd(cand));
                    weights = weights .* crowdW;
                    if ~isempty(RefNorm)
                        sec = sectorID(idx);
                        ref = RefNorm(max(1,min(sec,size(RefNorm,1))),:);
                        cheby = max(bsxfun(@rdivide,Yn(cand,:),max(ref,1e-6)),[],2);
                        weights = weights .* exp(-tauCheby*cheby);
                    end
                end

                if all(weights<=0)
                    weights = ones(size(weights));
                end
                weights = weights ./ sum(weights);
                Yteacher(idx,:) = weights' * candY;
            end
        end

        function sectorBest = pick_sector_elites(Y,Yn,goodIdx,sectorID,RefNorm,tauCheby)
            if isempty(RefNorm)
                sectorBest = [];
                return;
            end
            R = size(RefNorm,1);
            sectorBest = zeros(R,1);
            for r = 1:R
                cand = goodIdx(sectorID(goodIdx)==r);
                if isempty(cand)
                    cand = goodIdx;
                end
                if isempty(cand)
                    continue;
                end
                ref = max(RefNorm(r,:),1e-6);
                cheby = max(bsxfun(@rdivide,Yn(cand,:),ref),[],2);
                bias  = tauCheby * sum(Yn(cand,:),2);
                [~,bestIdx] = min(cheby + bias);
                sectorBest(r) = cand(bestIdx);
            end
        end

        function Ynew = relax_population(Y,Yteacher,zmin,goodMask,alpha,beta)
            N = size(Y,1);
            M = size(Y,2);

            alphaGood = 0.6 * alpha;
            alphaBad  = 1.2 * alpha;
            betaGood  = 0.5 * beta;
            betaBad   = 1.5 * beta;

            alphaVec = alphaBad * ones(N,1);
            alphaVec(goodMask) = alphaGood;
            betaVec = betaBad * ones(N,1);
            betaVec(goodMask) = betaGood;

            alphaMat = repmat(alphaVec,1,M);
            betaMat  = repmat(betaVec,1,M);
            zMat     = repmat(zmin,N,1);

            Ynew = (1 - alphaMat).*Y + alphaMat.*((1 - betaMat).*Yteacher + betaMat.*zMat);

            lower = min(Y,[],1) - 1e-8;
            upper = max(Y,[],1) + 1e-8;
            overMask = any(bsxfun(@lt,Ynew,lower),2) | any(bsxfun(@gt,Ynew,upper),2);
            Ynew(overMask,:) = (Y(overMask,:) + Yteacher(overMask,:))/2;
        end

        function [T,Ymean,Ystd] = learn_linear_mapping(Y,X,lambdaT)
            Ymean = mean(Y,1);
            Ystd  = std(Y,0,1);
            Ystd(Ystd < 1e-12) = 1;
            Yz = (Y - Ymean) ./ Ystd;
            M = size(Y,2);
            T = (Yz' * Yz + lambdaT * eye(M)) \ (Yz' * X);
        end

        function Xnew = map_linear(Yin,T,Ymean,Ystd,lb,ub)
            if isempty(T) || isempty(Ymean) || isempty(Ystd)
                Xnew = OS_DNS_Soft.mid_point(size(Yin,1),lb,ub);
                return;
            end
            Yz = (Yin - Ymean) ./ Ystd;
            Xraw = Yz * T;
            Xnew = OS_DNS_Soft.clamp(Xraw,lb,ub);
        end

        function Xnew = map_local(Yin,Yref,Xref,kLocal,lb,ub)
            if isempty(Yref) || isempty(Xref)
                Xnew = OS_DNS_Soft.mid_point(size(Yin,1),lb,ub);
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

            Xnew = OS_DNS_Soft.clamp(Xnew,lb,ub);
        end

        function Xnew = mid_point(N,lb,ub)
            [lbv,ubv] = OS_DNS_Soft.expand_bounds(lb,ub);
            mid = (lbv + ubv)/2;
            Xnew = repmat(mid,N,1);
        end

        function Xclamp = clamp(X,lb,ub)
            [lbv,ubv] = OS_DNS_Soft.expand_bounds(lb,ub,size(X,2));
            Xclamp = min(repmat(ubv,size(X,1),1), max(repmat(lbv,size(X,1),1), X));
        end

        function [lbv,ubv] = expand_bounds(lb,ub,D)
            if nargin < 3 || isempty(D)
                D = max(numel(lb),numel(ub));
                if D == 0
                    D = 1;
                end
            end
            if isscalar(lb)
                lbv = repmat(lb,1,D);
            else
                lbv = lb(:)';
                if numel(lbv) < D
                    lbv = [lbv repmat(lbv(end),1,D-numel(lbv))];
                elseif numel(lbv) > D
                    lbv = lbv(1:D);
                end
            end
            if isscalar(ub)
                ubv = repmat(ub,1,D);
            else
                ubv = ub(:)';
                if numel(ubv) < D
                    ubv = [ubv repmat(ubv(end),1,D-numel(ubv))];
                elseif numel(ubv) > D
                    ubv = ubv(1:D);
                end
            end
            if isempty(lbv)
                lbv = zeros(1,D);
            end
            if isempty(ubv)
                ubv = ones(1,D);
            end
        end

        function Xnew = restart_bad_solutions(Xnew,Xold,goodMask,gridInfo,restartRate,lb,ub)
            badIdx = find(~goodMask);
            if isempty(badIdx) || restartRate <= 0
                return;
            end

            crowd = gridInfo.crowd;
            [~,order] = sort(crowd(badIdx),'descend');
            nRestart = min(numel(badIdx), max(1, round(restartRate * numel(badIdx))));
            sel = badIdx(order(1:nRestart));

            goodIdx = find(goodMask);
            if isempty(goodIdx)
                goodIdx = 1:size(Xold,1);
            end

            [lbv,ubv] = OS_DNS_Soft.expand_bounds(lb,ub,size(Xold,2));
            span = max(ubv - lbv, 1e-9);

            for i = 1:numel(sel)
                anchor = Xold(goodIdx(randi(numel(goodIdx))),:);
                noise  = 0.1 * span .* randn(1,size(Xold,2));
                candidate = min(max(anchor + noise, lbv), ubv);
                Xnew(sel(i),:) = candidate;
            end
        end

        function Xnew = inject_diversity(Xnew,X,rate,Problem)
            n = size(Xnew,1);
            nOff = max(1,round(rate*n));
            nOff = min(nOff,floor(size(X,1)/2));
            if nOff <= 0
                return;
            end

            parentIdx = randi(size(X,1),1,2*nOff);
            parents   = X(parentIdx,:);
            mutants   = OperatorGAhalf(Problem,parents);
            replaceN  = min(size(mutants,1),n);
            if replaceN <= 0
                return;
            end
            replaceIdx = randperm(n,replaceN);
            Xnew(replaceIdx,:) = mutants(1:replaceN,:);
        end

        function PopOut = env_select_nsga2(PopIn,N)
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
                Yrange = max(max(Objs,[],1)-Ymin,1e-12);
                Yn = (Objs - Ymin) ./ Yrange;
                score = sum(Yn,2);
                [~,ord] = sort(score,'ascend');
                PopOut = PopIn(ord(1:N));
            end
        end
    end
end
