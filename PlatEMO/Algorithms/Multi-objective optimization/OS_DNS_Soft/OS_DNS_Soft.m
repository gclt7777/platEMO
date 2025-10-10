classdef OS_DNS_Soft < ALGORITHM
% <multi> <real/integer> <large/none>
% OS_DNS_Soft  Soft-Teacher + Full Relaxation with Linear Pre-Image
%
% 思路（不再二分 NS/DS）：
%   1) 目标空间对每个个体 y_i 构造“软教师目标” y_i^*（由 ε-支配+近邻+参考扇区加权）；
%   2) 全体松弛更新：Y_new = (1-α)Y + α * [ (1-β)Y* + β z_min ]；
%   3) 学线性预像 T：X ≈ zscore(Y)*T，用其将 Y_new → X_new；
%   4) 评估后用环境选择（HypE / RVEA / NSGA-II）得到下一代。
%
% 参数（Algorithm.ParameterSet）：
%   alpha        [0.40]  松弛步长（对 y→y* 的跟随强度）
%   beta         [0.15]  理想点牵引（对 zmin 的偏置）
%   lambdaT      [1e-6]  线性预像 T 的岭惩罚
%   tUpdateFreq  [1]     T 的更新频率（代）
%   reuseT       [false] 是否复用旧 T（false=每到频率就重学）
%   epsRate      [0.01]  ε-支配阈值（相对每维范围的比例 0.5~2% 推荐）
%   kNeighbor    [10]    软教师候选的近邻数（无 ε-老师时的兜底）
%   sigmaScale   [1.0]   高斯权重 σ 的系数（基于中位邻距）
%   tauCheby     [0.5]   切比雪夫权重的温度（越小越偏向沿参考向量更优者）
%   useRef       [1]     是否启用参考向量加权（0/1）
%   envMode      [1]     环境选择：1=HypE，2=RVEA(APD)，其他=NSGA-II
%   kLocalMap    [6]     局部预像的近邻数（>0 时启用局部非线性映射）
%   localMix     [0.5]   线性/局部预像的混合比例（0=仅线性，1=仅局部）
%   diversityRate[0.2]   注入多样化变异个体的比例（相对种群规模）
%
% 用法示例：
%   platemo('algorithm',@OS_DNS_Soft,'problem',@DTLZ2,'M',8,'D',100,'N',210,...
%           'maxFE',1e5,'envMode',1);   % HypE
%   platemo('algorithm',@OS_DNS_Soft,'problem',@DTLZ2,'M',15,'D',200,'N',270,...
%           'maxFE',1.5e5,'envMode',2); % RVEA

    methods
        function main(Algorithm,Problem)
            %% 参数
            [alpha,beta,lambdaT,tUpdateFreq,reuseT, ...
             epsRate,kNeighbor,sigmaScale,tauCheby,useRef,envMode, ...
             kLocalMap,localMix,diversityRate,refAdaptFreq,restartRate,gridDiv] = ...
             Algorithm.ParameterSet(0.40,0.15,1e-6,1,false, ...
                                    0.01, 10,       1.0,      0.5,     1,     1, ...
                                    6,    0.5,      0.2,      10,      0.15,   8);

            %% 初始化
            Population = Problem.Initialization();
            N = Problem.N;  D = Problem.D;  M = Problem.M; %#ok<NASGU>
            lb = Problem.lower;  ub = Problem.upper;

            Y = Population.objs;
            X = Population.decs;

            % 推迟到循环首轮再学习 T，避免重复学习
            T = []; Ymean = []; Ystd = [];
            if useRef
                [RefBase,~] = UniformPoint(max(N,Problem.M),Problem.M);
                RefVec = RefBase;
            else
                RefBase = [];
                RefVec  = [];
            end
            gen = 1;

            %% 进化主循环
            while Algorithm.NotTerminated(Population)
                % 1) 软教师集合（每个体一个 y*）
                [FrontNo,MaxFNo] = NDSort(Y, N);
                goodMask = FrontNo==1;
                if sum(goodMask) < max(2,ceil(0.2*N))
                    goodMask = FrontNo <= min(MaxFNo,2);
                end
                if useRef
                    if mod(gen-1, max(1,round(refAdaptFreq)))==0 || gen==1
                        refSource = Y(goodMask,:);
                        if isempty(refSource)
                            refSource = Y;
                        end
                        RefVec = OS_DNS_Soft.adapt_reference_vectors(refSource, RefBase);
                        RefVec = RefVec ./ max(1e-12,sqrt(sum(RefVec.^2,2)));
                    end
                else
                    RefVec = [];
                end
                zmin = min(Y,[],1);
                [Ystar,maskInfo] = OS_DNS_Soft.soft_teacher_targets(Y, goodMask, RefVec, epsRate, kNeighbor, sigmaScale, tauCheby, gridDiv, zmin);

                % 2) 全体松弛更新（含理想点牵引，带自适应步长）
                domRatio   = sum(Ystar <= Y + 1e-9, 2) ./ size(Y,2);
                alpha_vec  = alpha .* (0.2 + 0.8 .* domRatio);
                beta_vec   = beta  .* (0.1 + 0.9 .* domRatio);
                alpha_mat  = repmat(alpha_vec(:),1,size(Y,2));
                beta_mat   = repmat(beta_vec(:),1,size(Y,2));
                zmin_mat   = repmat(zmin,size(Y,1),1);
                Y_new      = (1 - alpha_mat).*Y + alpha_mat.*((1 - beta_mat).*Ystar + beta_mat.*zmin_mat);

                % 3) 线性预像 Y->X（EAGO风格）
                if gen==1 || (~reuseT && mod(gen-1,tUpdateFreq)==0)
                    [T, Ymean, Ystd] = OS_DNS_Soft.learnT_linear(Y, X, lambdaT);
                end
                X_linear = OS_DNS_Soft.mapY2X_linear(Y_new, T, lb, ub, Ymean, Ystd);
                if kLocalMap > 0 && localMix > 0
                    X_local  = OS_DNS_Soft.mapY2X_local(Y_new, Y, X, kLocalMap, lb, ub);
                    mixRatio = min(max(localMix,0),1);
                    X_new    = (1 - mixRatio).*X_linear + mixRatio.*X_local;
                else
                    X_new = X_linear;
                end
                X_new = X_new + 1e-12*randn(size(X_new));   % 破并列的极微扰动

                if restartRate > 0
                    X_new = OS_DNS_Soft.restart_bad_solutions(X_new, X, lb, ub, maskInfo.badMask, maskInfo.gridCrowd, restartRate);
                end

                if diversityRate > 0 && size(X_new,1) > 1 && size(X,1) > 1
                    nOff = max(1, round(diversityRate * size(X_new,1)));
                    maxPairs = floor(size(X,1)/2);
                    nOff = min(nOff, maxPairs);
                    if nOff > 0
                        parentIdx = randi(size(X,1),1,2*nOff);
                        parents   = X(parentIdx,:);
                        mutants   = OperatorGAhalf(Problem, parents);
                        replaceN  = min(size(mutants,1), size(X_new,1));
                        if replaceN > 0
                            replaceIdx = randperm(size(X_new,1), replaceN);
                            X_new(replaceIdx,:) = mutants(1:replaceN,:);
                        end
                    end
                end

                % 4) 评估与环境选择
                Offspring = Problem.Evaluation(X_new);
                PopAll    = [Population, Offspring];

                switch envMode
                    case 1
                        [Population,FrontNo,CrowdDis] = env_select_hype(PopAll, N); %#ok<ASGLU>
                    case 2
                        % RVEA 的 APD 惩罚系数可随进度变化，简单用 FE/maxFE 近似
                        theta = 1;
                        try
                            if isprop(Algorithm,'FE') && isprop(Problem,'maxFE')
                                theta = min(2, max(0.5, 2*(Algorithm.FE / max(1,Problem.maxFE))));
                            end
                        catch, theta = 1; end
                        [Population,FrontNo,CrowdDis] = env_select_rvea(PopAll, N, theta); %#ok<ASGLU>
                    otherwise
                        [Population,FrontNo,CrowdDis] = OS_DNS_Soft.envSelect_NSGA2(PopAll, N); %#ok<ASGLU>
                end

                % 5) 更新当前 Y/X
                Y = Population.objs;
                X = Population.decs;
                gen = gen + 1;
            end
        end
    end

    %% ======== 辅助静态方法 ========
    methods (Static, Access = private)
        function [T, Ymean, Ystd] = learnT_linear(Y, X, lambdaT)
            Ymean = mean(Y, 1);
            Ystd  = std(Y, 0, 1);  Ystd(Ystd < 1e-12) = 1;
            Yz = (Y - Ymean) ./ Ystd;       % N x M
            M = size(Y,2);
            T = (Yz' * Yz + lambdaT * eye(M)) \ (Yz' * X);  % M x D
        end

        function Xnew = mapY2X_linear(Yin, T, lb, ub, Ymean, Ystd)
            Yz = (Yin - Ymean) ./ Ystd;
            Xnew = Yz * T;                  % N x D
            [N, D] = size(Xnew);
            if isscalar(lb), lb = repmat(lb, 1, D); end
            if isscalar(ub), ub = repmat(ub, 1, D); end
            if iscolumn(lb), lb = lb'; end
            if iscolumn(ub), ub = ub'; end
            if numel(lb) ~= D, lb = repmat(lb(1), 1, D); end
            if numel(ub) ~= D, ub = repmat(ub(1), 1, D); end
            Xnew = min(repmat(ub,N,1), max(repmat(lb,N,1), Xnew));
        end

        function Xnew = mapY2X_local(Yin, Yref, Xref, kLocal, lb, ub)
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
                mid = (lbv + ubv) / 2;
                Xnew = repmat(mid, size(Yin,1), 1);
                return;
            end
            kLocal = max(1, min(kLocal, size(Yref,1)));
            try
                dist = pdist2(Yin, Yref);
            catch
                dist = sqrt(max(0, sum(Yin.^2,2) + sum(Yref.^2,2)' - 2*(Yin*Yref')));
            end
            [~, order] = sort(dist, 2);
            idx = order(:,1:kLocal);
            weights = zeros(size(idx));
            for i = 1:size(idx,1)
                di = dist(i, idx(i,:));
                if all(di < 1e-12)
                    w = zeros(1,kLocal);
                    w(di<1e-12) = 1;
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
                w = weights(i,:)';
                Xnew(i,:) = (w' * neighX);
            end
            [N,D] = size(Xnew);
            if isscalar(lb), lb = repmat(lb, 1, D); end
            if isscalar(ub), ub = repmat(ub, 1, D); end
            if iscolumn(lb), lb = lb'; end
            if iscolumn(ub), ub = ub'; end
            if numel(lb) ~= D, lb = repmat(lb(1), 1, D); end
            if numel(ub) ~= D, ub = repmat(ub(1), 1, D); end
            Xnew = min(repmat(ub,N,1), max(repmat(lb,N,1), Xnew));
        end

        function [Ystar,info] = soft_teacher_targets(Y, goodMask, RefVec, epsRate, kNeighbor, sigmaScale, tauCheby, gridDiv, zmin)
            N = size(Y,1);
            Ystar = Y;
            kNeighbor = max(1, round(kNeighbor));
            sigmaScale = max(0, sigmaScale);
            if nargin < 9 || isempty(zmin)
                zmin = min(Y,[],1);
            end
            if nargin < 8 || isempty(gridDiv)
                gridDiv = 8;
            end
            goodMask = logical(goodMask(:));
            if ~any(goodMask)
                [~,best] = min(sum(Y,2));
                goodMask(best) = true;
            end
            badMask = ~goodMask;

            Ymin = min(Y,[],1);
            Ymax = max(Y,[],1);
            Yrange = max(Ymax - Ymin, 1e-12);
            epsVec = epsRate .* Yrange;
            Yn = (Y - Ymin) ./ Yrange;
            [gridCrowd,gridID] = OS_DNS_Soft.grid_crowding(Yn, gridDiv);

            goodIdx = find(goodMask);
            badIdx  = find(badMask);
            Ygood   = Y(goodIdx,:);
            Ybad    = Y(badIdx,:);

            if isempty(Ygood)
                Ygood = Y;
                goodIdx = (1:N)';
            end

            % sector assignment via reference vectors
            if ~isempty(RefVec)
                RefNorm = RefVec ./ max(1e-12,sqrt(sum(RefVec.^2,2)));
                YnDir   = Yn ./ max(1e-12,sqrt(sum(Yn.^2,2)));
                simVal  = YnDir * RefNorm';
                [~,sectorIdx] = max(simVal,[],2);
            else
                sectorIdx = ones(N,1);
                RefNorm = [];
            end

            % choose representative good solutions per sector (Chebyshev metric)
            sectorBest = zeros(size(RefVec,1),1);
            if ~isempty(RefVec)
                for r = 1:size(RefVec,1)
                    cand = goodIdx(sectorIdx(goodIdx)==r);
                    if isempty(cand)
                        cand = goodIdx;
                    end
                    if isempty(cand)
                        continue;
                    end
                    candNorm = Yn(cand,:);
                    ref = max(RefNorm(r,:),1e-6);
                    cheby = max((candNorm+1e-9)./ref,[],2) + tauCheby*sum(candNorm,2);
                    [~,bestLocal] = min(cheby);
                    sectorBest(r) = cand(bestLocal);
                end
            end

            % good solutions target: slight pull toward sector exemplar or ideal point
            if ~isempty(goodIdx)
                goodTarget = Y(goodIdx,:);
                if ~isempty(RefVec)
                    for i = 1:length(goodIdx)
                        g = goodIdx(i);
                        sector = sectorIdx(g);
                        exemplar = g;
                        if sector <= numel(sectorBest) && sectorBest(sector) ~= 0
                            exemplar = sectorBest(sector);
                        end
                        if exemplar == g
                            mix = min(max(tauCheby,0),0.5);
                            goodTarget(i,:) = (1 - mix).*Y(g,:) + mix.*zmin;
                        else
                            mix = min(max(tauCheby,0),0.7);
                            goodTarget(i,:) = (1 - mix).*Y(g,:) + mix.*Y(exemplar,:);
                        end
                    end
                else
                    mix = min(max(tauCheby,0),0.5);
                    goodTarget = (1 - mix).*Y(goodIdx,:) + mix.*repmat(zmin,length(goodIdx),1);
                end
                Ystar(goodIdx,:) = goodTarget;
            end

            if ~isempty(badIdx)
                try
                    dist = pdist2(Ybad, Ygood);
                catch
                    dist = sqrt(max(0, sum(Ybad.^2,2) + sum(Ygood.^2,2)' - 2*(Ybad*Ygood')));
                end
                if isempty(dist)
                    dist = zeros(length(badIdx),1);
                end
                sigBase = median(dist(:));
                if ~isfinite(sigBase) || sigBase <= 0
                    sigBase = 1;
                end
                for ii = 1:length(badIdx)
                    idx = badIdx(ii);
                    candMask = all(bsxfun(@le, Ygood, Y(idx,:) + epsVec + 1e-9),2);
                    cand = goodIdx(candMask);
                    if isempty(cand)
                        if ~isempty(RefVec)
                            sec = sectorIdx(idx);
                            if sec <= numel(sectorBest) && sectorBest(sec) ~= 0
                                cand = sectorBest(sec);
                            end
                        end
                    end
                    if isempty(cand)
                        [~,order] = sort(dist(ii,:),'ascend');
                        kn = min(kNeighbor,length(order));
                        cand = goodIdx(order(1:kn));
                    end
                    if isempty(cand)
                        cand = goodIdx(randi(length(goodIdx)));
                    end
                    cand = cand(:);
                    candY = Y(cand,:);
                    if numel(cand) > 1
                        d = sqrt(sum((candY - Y(idx,:)).^2,2));
                        sigma = max(sigmaScale*sigBase,1e-9);
                        w = exp(-d.^2./(2*sigma^2));
                        if ~isempty(RefVec)
                            sec = sectorIdx(idx);
                            ref = max(1e-6,RefNorm(max(1,min(sec,size(RefNorm,1))),:));
                            cheby = max(((Yn(cand,:))+1e-9)./ref,[],2);
                            tau = min(max(tauCheby,0),1);
                            w = w .* exp(-tau*cheby);
                        end
                    else
                        w = 1;
                    end
                    if all(w <= 0)
                        w = ones(size(w));
                    end
                    w = w ./ sum(w);
                    Ystar(idx,:) = w' * candY;
                end
            end

            info.goodMask = goodMask;
            info.badMask  = badMask;
            info.gridCrowd = gridCrowd;
            info.gridID    = gridID;
        end

        function [crowd,cellID] = grid_crowding(Yn, gridDiv)
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

        function Xnew = restart_bad_solutions(Xnew, Xold, lb, ub, badMask, gridCrowd, restartRate)
            if ~any(badMask) || restartRate <= 0
                return;
            end
            idxBad = find(badMask);
            if isempty(gridCrowd)
                badCrowd = ones(size(idxBad));
            else
                badCrowd = gridCrowd(idxBad);
            end
            nRestart = min(numel(idxBad), max(0, round(restartRate * numel(idxBad))));
            if nRestart <= 0
                return;
            end
            [~,order] = sort(badCrowd,'descend');
            idxBad = idxBad(order(1:nRestart));

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

            for i = 1:length(idxBad)
                teacher = Xold(goodIdx(randi(length(goodIdx))),:);
                noise = 0.1 * span .* randn(1,size(Xnew,2));
                candidate = teacher + noise;
                candidate = min(max(candidate, lbv), ubv);
                Xnew(idxBad(i),:) = candidate;
            end
        end

        function Ref = adapt_reference_vectors(PopObj, RefBase)
            if isempty(RefBase)
                Ref = [];
                return;
            end
            range = max(PopObj,[],1) - min(PopObj,[],1);
            Ref = RefBase .* repmat(range,size(RefBase,1),1);
        end

        function [PopOut,FrontNo,CrowdDis] = envSelect_NSGA2(PopIn, N)
            try
                Objs = PopIn.objs;
                [FrontNo,MaxFNo] = NDSort(Objs, N);
                Next = FrontNo < MaxFNo;
                CrowdDis = CrowdingDistance(Objs, FrontNo);
                Last = find(FrontNo==MaxFNo);
                [~,rank] = sort(CrowdDis(Last),'descend');
                Next(Last(rank(1:N-sum(Next)))) = true;
                PopOut = PopIn(Next);
                FrontNo = FrontNo(Next);
                CrowdDis = CrowdDis(Next);
            catch
                % 兜底：归一化打分
                Objs = PopIn.objs;
                Ymin = min(Objs,[],1); Ymax = max(Objs,[],1);
                rng  = max(Ymax - Ymin, 1e-12);
                Yn   = (Objs - Ymin) ./ rng;
                score = sum(Yn,2);
                [~,ord] = sort(score,'ascend');
                PopOut = PopIn(ord(1:N));
                FrontNo = ones(N,1);
                CrowdDis = inf(N,1);
            end
        end
    end
end
