classdef PHLMEA < ALGORITHM
% <large/real/integer/binary> <many> <constrained/none>
% Population-hierarchical learning based EA for large-scale MaOPs

%------------------------------- Reference --------------------------------
% X. Zhang, Y. Tian, R. Cheng, and Y. Jin, A decision variable clustering
% based evolutionary algorithm for large-scale many-objective optimization.
% IEEE Transactions on Evolutionary Computation, 2018, 22(1): 97-112.
% L. Zhao, Y. Tian, R. Cheng, and X. Zhang, A population hierarchical-based
% evolutionary algorithm for large-scale many-objective optimization.
% Information Sciences, 2021, 569: 84-110.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [ratioLow,archiveFactor,clusterAlpha,varFreq,varKeep] = ...
                Algorithm.ParameterSet(0.5,1.5,1.0,0,0.3);

            %% Initial population and archive
            Population = Problem.Initialization();
            archiveSize = max(Problem.N, round(archiveFactor*Problem.N));

            [P_low,P_high] = SplitPopulationLayers(Population,ratioLow);
            Archive       = UpdateArchive(Population,SOLUTION.empty,archiveSize);

            activeIndex = 1:Problem.D;
            generation  = 0;

            %% Optimization loop
            while Algorithm.NotTerminated(Population)
                generation = generation + 1;

                Off_low  = ReproduceLowLayer(P_low,Archive,P_high,Problem,activeIndex);
                Off_high = ReproduceHighLayer(P_high,Archive,Problem,activeIndex);

                Union = [Population,Off_low,Off_high,Archive];
                Population = ClusteredEnvironmentalSelection(Union,Problem.N,clusterAlpha);

                [P_low,P_high] = SplitPopulationLayers(Population,ratioLow);
                Archive        = UpdateArchive(Population,Archive,archiveSize);

                if varFreq > 0 && mod(generation,varFreq) == 0
                    activeIndex = IdentifyActiveVariables([Population,Archive],Problem,varKeep);
                end
            end
        end
    end
end

function [P_low,P_high] = SplitPopulationLayers(Population,ratioLow)
% Split the population into low- and high-fitness layers based on rank and crowding distance.

    if isempty(Population)
        P_low  = Population;
        P_high = Population;
        return;
    end

    PopObj  = Population.objs;
    PopCons = Population.cons;
    if isempty(PopCons)
        PopCons = [];
    end

    [FrontNo,~] = NDSort(PopObj,PopCons,numel(Population));
    CrowdDis    = CrowdingDistance(PopObj,FrontNo);

    scores = [FrontNo',-CrowdDis'];
    [~,order] = sortrows(scores,[1 2]);

    cutPoint = max(1,min(numel(Population),round(ratioLow*numel(Population))));
    P_low    = Population(order(1:cutPoint));
    P_high   = Population(order(cutPoint+1:end));
end

function Archive = UpdateArchive(Population,Archive,maxArchive)
% Update the external archive using non-dominated sorting and crowding.

    Combined = [Archive,Population];
    if isempty(Combined)
        Archive = Combined;
        return;
    end

    Decs = cat(1,Combined.decs);
    DecsRounded = round(Decs,8);
    [~,uniqueIdx] = unique(DecsRounded,'rows','stable');
    Combined = Combined(uniqueIdx);

    PopObj  = Combined.objs;
    PopCons = Combined.cons;
    if isempty(PopCons)
        PopCons = [];
    end

    [FrontNo,MaxFront] = NDSort(PopObj,PopCons,maxArchive);
    Next    = FrontNo < MaxFront;
    Archive = Combined(Next);

    Last  = find(FrontNo == MaxFront);
    picks = maxArchive - sum(Next);
    if picks > 0 && ~isempty(Last)
        CrowdDis = CrowdingDistance(PopObj,FrontNo);
        [~,rank] = sort(CrowdDis(Last),'descend');
        picks    = min(picks,numel(rank));
        Archive  = [Archive,Combined(Last(rank(1:picks)))];
    end
end

function Population = ClusteredEnvironmentalSelection(Population,N,clusterAlpha)
% Environmental selection based on non-dominated sorting and clustering.

    if isempty(Population)
        return;
    end

    PopObj  = Population.objs;
    PopCons = Population.cons;
    if isempty(PopCons)
        PopCons = [];
    end

    [FrontNo,MaxFront] = NDSort(PopObj,PopCons,N);
    Next = FrontNo < MaxFront;

    Last     = find(FrontNo == MaxFront);
    Remain   = N - sum(Next);
    Selected = false(1,numel(Population));
    Selected(Next) = true;

    if Remain > 0 && ~isempty(Last)
        frontObj = PopObj(Last,:);
        normObj  = NormaliseObjectives(frontObj);

        clusterCount = max(1,min(numel(Last),round(clusterAlpha*size(frontObj,2))));
        clusterCount = min(clusterCount,Remain);

        chosen = ClusterSelect(normObj,clusterCount,Remain);
        Selected(Last(chosen)) = true;
    end

    Population = Population(Selected);
end

function Offspring = ReproduceLowLayer(P_low,Archive,P_high,Problem,activeIndex)
% Reproduction for the low-fitness layer guided by archive and high layer.

    if isempty(P_low)
        Offspring = P_low;
        return;
    end

    n = numel(P_low);
    OffDec = cat(1,P_low.decs);
    base   = OffDec;
    lower  = repmat(Problem.lower,n,1);
    upper  = repmat(Problem.upper,n,1);

    ArchDec = [];
    if ~isempty(Archive)
        ArchDec = cat(1,Archive.decs);
    end
    if isempty(ArchDec)
        ArchDec = base;
    end

    HighDec = [];
    if ~isempty(P_high)
        HighDec = cat(1,P_high.decs);
    end
    if isempty(HighDec)
        HighDec = base;
    end

    for i = 1:n
        a = base(i,:);
        learn = ArchDec(randi(size(ArchDec,1)),:);
        diff  = HighDec(randi(size(HighDec,1)),:);

        F  = 0.5 + rand*0.5;
        CR = 0.5 + rand*0.5;
        mutant = a + F*(learn - diff);

        crossMask = rand(1,Problem.D) < CR;
        if ~isempty(activeIndex)
            crossMask(activeIndex(randi(numel(activeIndex)))) = true;
        else
            crossMask(randi(Problem.D)) = true;
        end

        trial = a;
        trial(crossMask) = mutant(crossMask);
        OffDec(i,:) = trial;
    end

    OffDec = min(max(OffDec,lower),upper);
    Offspring = Problem.Evaluation(OffDec);
end

function Offspring = ReproduceHighLayer(P_high,Archive,Problem,activeIndex)
% Reproduction for the high-fitness layer focusing on archive exploitation.

    if isempty(P_high)
        Offspring = P_high;
        return;
    end

    n = numel(P_high);
    OffDec = cat(1,P_high.decs);
    base   = OffDec;
    lower  = repmat(Problem.lower,n,1);
    upper  = repmat(Problem.upper,n,1);

    if isempty(Archive)
        ArchDec = base;
    else
        ArchDec = cat(1,Archive.decs);
    end

    for i = 1:n
        idx = randperm(size(ArchDec,1),min(2,size(ArchDec,1)));
        if numel(idx) == 1
            learn1 = ArchDec(idx,:);
            learn2 = base(randi(n),:);
        else
            learn1 = ArchDec(idx(1),:);
            learn2 = ArchDec(idx(2),:);
        end

        a  = base(i,:);
        F  = 0.4 + rand*0.4;
        CR = 0.6 + rand*0.3;
        mutant = a + F*(learn1 - learn2);

        crossMask = rand(1,Problem.D) < CR;
        if ~isempty(activeIndex)
            crossMask(activeIndex(randi(numel(activeIndex)))) = true;
        else
            crossMask(randi(Problem.D)) = true;
        end

        trial = a;
        trial(crossMask) = mutant(crossMask);
        OffDec(i,:) = trial;
    end

    OffDec = min(max(OffDec,lower),upper);
    Offspring = Problem.Evaluation(OffDec);
end

function activeIndex = IdentifyActiveVariables(Individuals,Problem,keepRatio)
% Estimate active decision variables using normalised standard deviation.

    if isempty(Individuals)
        activeIndex = 1:Problem.D;
        return;
    end

    Decs = cat(1,Individuals.decs);
    sigma = std(Decs,0,1);
    span  = Problem.upper - Problem.lower;
    span(span == 0) = 1;
    contribution = sigma ./ span;

    activeCount = max(1,round(keepRatio*Problem.D));
    [~,order] = sort(contribution,'descend');
    activeIndex = order(1:activeCount);
end

function normObj = NormaliseObjectives(PopObj)
% Normalise objective values to [0,1].

    fmin = min(PopObj,[],1);
    fmax = max(PopObj,[],1);
    span = fmax - fmin;
    span(span <= 0) = 1;
    normObj = (PopObj - fmin) ./ span;
end

function chosen = ClusterSelect(normObj,clusterCount,remain)
% Select individuals from the last front via clustering inspired grouping.

    [N,~] = size(normObj);
    if N <= remain
        chosen = 1:N;
        return;
    end

    scores = sum(normObj,2);
    [~,bestIdx] = min(scores);
    centers = normObj(bestIdx,:);
    centerId = bestIdx;

    candidates = setdiff(1:N,bestIdx);
    while size(centers,1) < clusterCount && ~isempty(candidates)
        dist = MinDistance(normObj(candidates,:),centers);
        [~,pos] = max(dist);
        centers = [centers;normObj(candidates(pos),:)]; %#ok<AGROW>
        centerId = [centerId,candidates(pos)]; %#ok<AGROW>
        candidates(pos) = [];
    end

    assign = zeros(1,N);
    for i = 1:N
        [~,assign(i)] = min(sum((centers - normObj(i,:)).^2,2));
    end

    chosen = [];
    for c = 1:size(centers,1)
        members = find(assign == c);
        if isempty(members)
            continue;
        end
        [~,idx] = min(scores(members));
        chosen = [chosen,members(idx)]; %#ok<AGROW>
        if numel(chosen) >= remain
            chosen = chosen(1:remain);
            return;
        end
    end

    available = setdiff(1:N,chosen);
    while numel(chosen) < remain && ~isempty(available)
        dist = MinDistance(normObj(available,:),normObj(chosen,:));
        [~,pos] = max(dist);
        chosen = [chosen,available(pos)]; %#ok<AGROW>
        available(pos) = [];
    end
end

function dist = MinDistance(points,centers)
% Minimum squared distance from each point to existing centers.

    if isempty(centers)
        dist = inf(size(points,1),1);
        return;
    end

    dist = inf(size(points,1),1);
    for k = 1:size(centers,1)
        diff = points - centers(k,:);
        d = sum(diff.^2,2);
        dist = min(dist,d);
    end
end

function CrowdDis = CrowdingDistance(PopObj,FrontNo)
% Calculate the crowding distance of each solution front by front.

    [N,M]  = size(PopObj);
    CrowdDis = zeros(1,N);
    Fronts = setdiff(unique(FrontNo),inf);
    for f = 1 : numel(Fronts)
        Front = find(FrontNo == Fronts(f));
        if numel(Front) <= 2
            CrowdDis(Front) = inf;
            continue;
        end
        Fmax = max(PopObj(Front,:),[],1);
        Fmin = min(PopObj(Front,:),[],1);
        span = Fmax - Fmin;
        span(span == 0) = 1;
        for i = 1:M
            [~,rank] = sort(PopObj(Front,i));
            CrowdDis(Front(rank(1)))   = inf;
            CrowdDis(Front(rank(end))) = inf;
            for j = 2:length(rank)-1
                CrowdDis(Front(rank(j))) = CrowdDis(Front(rank(j))) + ...
                    (PopObj(Front(rank(j+1)),i) - PopObj(Front(rank(j-1)),i)) / span(i);
            end
        end
    end
end
