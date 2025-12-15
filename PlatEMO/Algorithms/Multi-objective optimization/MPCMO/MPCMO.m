classdef MPCMO < ALGORITHM
% <many> <real/integer/binary/permutation> <constrained/none>
% Multi-population co-evolutionary algorithm for MaOPs

%------------------------------- Reference --------------------------------
% C. He, Q. Zhang, Y. Tian, and J. Xiao, A multi-population co-evolutionary
% evolutionary algorithm for many-objective optimization. IEEE Transactions
% on Evolutionary Computation, 2018, 22(2): 224-239.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [numGroups,migrationPeriod,migrationFraction,refMode,globalSelectPeriod] = ...
                Algorithm.ParameterSet(5,5,0.2,1,0);

            numGroups          = max(1,round(numGroups));
            migrationPeriod    = max(0,round(migrationPeriod));
            migrationFraction  = max(0,migrationFraction);
            refMode            = refMode > 0;
            globalSelectPeriod = max(0,round(globalSelectPeriod));

            %% Initialise population and subpopulations
            Population = Problem.Initialization();
            totalSize  = numel(Population);
            if totalSize == 0
                return;
            end

            numGroups = min(numGroups,totalSize);
            while numGroups > 1 && totalSize/numGroups < 2
                numGroups = numGroups - 1;
            end

            [SubPops,subSizes,RefVectors] = InitialiseSubpopulations(Population,numGroups,refMode,Problem);
            Population = CombineSubpopulations(SubPops);

            generation = 0;

            %% Optimization loop
            while Algorithm.NotTerminated(Population)
                generation = generation + 1;

                % Evolve each subpopulation independently
                for g = 1:numGroups
                    SubPops{g} = EvolveSubpopulation(Problem,SubPops{g},subSizes(g));
                end

                % Periodic migration between neighbouring subpopulations
                if migrationFraction > 0 && migrationPeriod > 0 && mod(generation,migrationPeriod) == 0
                    SubPops = PerformMigration(SubPops,subSizes,migrationFraction);
                end

                Population = CombineSubpopulations(SubPops);

                % Optional global environmental selection and redistribution
                if globalSelectPeriod > 0 && mod(generation,globalSelectPeriod) == 0
                    [Population,~,~] = NSGAIIEnvironmentalSelection(Population,Problem.N);
                    [SubPops,RefVectors] = RedistributeSubpopulations(Population,subSizes,refMode,RefVectors,Problem);
                    Population = CombineSubpopulations(SubPops);
                end
            end
        end
    end
end

function SubPop = EvolveSubpopulation(Problem,SubPop,targetSize)
% Evolve a single subpopulation using GA reproduction and NSGA-II selection.

    if isempty(SubPop)
        SubPop = SOLUTION.empty;
        return;
    end

    subSize = numel(SubPop);
    [FrontNo,~] = NDSort(SubPop.objs,SubPop.cons,subSize);
    CrowdDis    = CrowdingDistance(SubPop.objs,FrontNo);

    if subSize > 1
        MatingPool = TournamentSelection(2,2*subSize,FrontNo,-CrowdDis);
        Offspring  = OperatorGA(Problem,SubPop(MatingPool));
    else
        Offspring  = OperatorGA(Problem,repmat(SubPop,1,2));
    end

    SubPop = NSGAIIEnvironmentalSelection([SubPop,Offspring],targetSize);
end

function SubPops = PerformMigration(SubPops,subSizes,migrationFraction)
% Perform ring migration by sending elite individuals to the next group.

    numGroups = numel(SubPops);
    if numGroups <= 1
        return;
    end

    migrants = cell(1,numGroups);
    for g = 1:numGroups
        subSize = numel(SubPops{g});
        if subSize == 0
            migrants{g} = SOLUTION.empty;
            continue;
        end
        count = min(subSize,max(1,round(migrationFraction*subSizes(g))));
        migrants{g} = SelectMigrants(SubPops{g},count);
    end

    for g = 1:numGroups
        target = mod(g,numGroups) + 1;
        if isempty(migrants{g})
            continue;
        end
        SubPops{target} = [SubPops{target},migrants{g}];
    end

    for g = 1:numGroups
        desired = subSizes(g);
        if numel(SubPops{g}) > desired
            SubPops{g} = NSGAIIEnvironmentalSelection(SubPops{g},desired);
        end
    end
end

function migrants = SelectMigrants(SubPop,count)
% Select elite individuals within a subpopulation for migration.

    if count <= 0 || isempty(SubPop)
        migrants = SOLUTION.empty;
        return;
    end

    subSize = numel(SubPop);
    [FrontNo,~] = NDSort(SubPop.objs,SubPop.cons,subSize);
    CrowdDis    = CrowdingDistance(SubPop.objs,FrontNo);

    ranking = [(1:subSize)', FrontNo(:), -CrowdDis(:)];
    ranking = sortrows(ranking,[2 3]);
    selected = ranking(1:min(count,size(ranking,1)),1);
    migrants = SubPop(selected);
end

function [SubPops,subSizes,RefVectors] = InitialiseSubpopulations(Population,numGroups,refMode,Problem)
% Create the initial set of subpopulations based on reference vectors or random splits.

    totalSize = numel(Population);
    baseSize  = floor(totalSize/numGroups);
    subSizes  = baseSize*ones(1,numGroups);
    remainder = totalSize - baseSize*numGroups;
    subSizes(1:remainder) = subSizes(1:remainder) + 1;

    if refMode
        [RefVectors,~] = UniformPoint(numGroups,Problem.M);
        RefVectors = NormaliseVectors(RefVectors);
        association = AssociateToReferences(Population.objs,RefVectors);
    else
        RefVectors  = [];
        association = ones(totalSize,1);
    end

    assigned = false(totalSize,1);
    SubPops  = cell(1,numGroups);

    for g = 1:numGroups
        need = subSizes(g);
        if refMode
            cand = find(association == g & ~assigned);
        else
            cand = find(~assigned);
        end
        if isempty(cand)
            SubPops{g} = SOLUTION.empty;
            continue;
        end
        take = min(need,numel(cand));
        idx  = cand(1:take);
        SubPops{g} = Population(idx);
        assigned(idx) = true;
    end

    remaining = find(~assigned);
    if ~isempty(remaining)
        remaining = remaining(randperm(numel(remaining)));
        ptr = 1;
        for g = 1:numGroups
            need = subSizes(g) - numel(SubPops{g});
            if need <= 0
                continue;
            end
            take = min(need,numel(remaining) - ptr + 1);
            if take <= 0
                continue;
            end
            idx = remaining(ptr:ptr+take-1);
            SubPops{g} = [SubPops{g},Population(idx)]; %#ok<AGROW>
            ptr = ptr + take;
        end
    end

    for g = 1:numGroups
        if numel(SubPops{g}) < subSizes(g)
            deficit = subSizes(g) - numel(SubPops{g});
            filler  = randperm(totalSize,deficit);
            SubPops{g} = [SubPops{g},Population(filler)]; %#ok<AGROW>
        elseif numel(SubPops{g}) > subSizes(g)
            SubPops{g} = SubPops{g}(1:subSizes(g));
        end
    end
end

function [SubPops,RefVectors] = RedistributeSubpopulations(Population,subSizes,refMode,RefVectors,Problem)
% Redistribute individuals into subpopulations after global selection.

    numGroups = numel(subSizes);
    if refMode && (isempty(RefVectors) || size(RefVectors,1) ~= numGroups)
        [RefVectors,~] = UniformPoint(numGroups,Problem.M);
        RefVectors = NormaliseVectors(RefVectors);
    end

    if refMode
        association = AssociateToReferences(Population.objs,RefVectors);
    else
        association = ones(numel(Population),1);
    end

    assigned = false(numel(Population),1);
    SubPops  = cell(1,numGroups);

    for g = 1:numGroups
        need = subSizes(g);
        if refMode
            cand = find(association == g & ~assigned);
        else
            cand = find(~assigned);
        end
        take = min(need,numel(cand));
        if take > 0
            idx = cand(1:take);
            SubPops{g} = Population(idx);
            assigned(idx) = true;
        else
            SubPops{g} = SOLUTION.empty;
        end
    end

    remaining = find(~assigned);
    if ~isempty(remaining)
        remaining = remaining(randperm(numel(remaining)));
        ptr = 1;
        for g = 1:numGroups
            need = subSizes(g) - numel(SubPops{g});
            if need <= 0
                continue;
            end
            take = min(need,numel(remaining) - ptr + 1);
            if take <= 0
                continue;
            end
            idx = remaining(ptr:ptr+take-1);
            SubPops{g} = [SubPops{g},Population(idx)]; %#ok<AGROW>
            ptr = ptr + take;
        end
    end

    for g = 1:numGroups
        if numel(SubPops{g}) < subSizes(g)
            deficit = subSizes(g) - numel(SubPops{g});
            filler  = randperm(numel(Population),deficit);
            SubPops{g} = [SubPops{g},Population(filler)]; %#ok<AGROW>
        elseif numel(SubPops{g}) > subSizes(g)
            SubPops{g} = SubPops{g}(1:subSizes(g));
        end
    end
end

function Population = CombineSubpopulations(SubPops)
% Combine all subpopulations into a single population array.

    if isempty(SubPops)
        Population = SOLUTION.empty;
    else
        Population = [SubPops{:}];
    end
end

function [Population,FrontNo,CrowdDis] = NSGAIIEnvironmentalSelection(Population,N)
% Local copy of NSGA-II environmental selection to avoid name conflicts.

    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;

    CrowdDis = CrowdingDistance(Population.objs,FrontNo);

    Last = find(FrontNo == MaxFNo);
    if ~isempty(Last)
        [~,Rank] = sort(CrowdDis(Last),'descend');
        picks = min(numel(Last), max(0, N - sum(Next)));
        if picks > 0
            Next(Last(Rank(1:picks))) = true;
        end
    end

    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end

function CrowdDis = CrowdingDistance(PopObj,FrontNo)
% Calculate the crowding distance of each solution in all fronts.

    [N,M]    = size(PopObj);
    CrowdDis = zeros(1,N);
    Fronts   = setdiff(unique(FrontNo),inf);
    for f = 1 : numel(Fronts)
        Front = find(FrontNo == Fronts(f));
        if isempty(Front)
            continue;
        end
        Fmax = max(PopObj(Front,:),[],1);
        Fmin = min(PopObj(Front,:),[],1);
        for i = 1 : M
            [~,Rank] = sortrows(PopObj(Front,i));
            CrowdDis(Front(Rank(1)))   = inf;
            CrowdDis(Front(Rank(end))) = inf;
            span = Fmax(i) - Fmin(i);
            if span <= 0
                continue;
            end
            for j = 2 : length(Front)-1
                CrowdDis(Front(Rank(j))) = CrowdDis(Front(Rank(j))) + ...
                    (PopObj(Front(Rank(j+1)),i) - PopObj(Front(Rank(j-1)),i)) / span;
            end
        end
    end
end

function [association,angles] = AssociateToReferences(PopObj,RefVectors)
% Associate each solution with the closest reference vector by angle.

    if isempty(PopObj)
        association = zeros(0,1);
        angles = zeros(0,1);
        return;
    end

    PopObj(~isfinite(PopObj)) = 0;
    shift = min(PopObj,[],1);
    PopObj = PopObj - shift;
    PopObj = max(PopObj,0);
    norms = sqrt(sum(PopObj.^2,2));
    zeroMask = norms <= eps;
    norms(zeroMask) = 1;
    normalisedObj = PopObj./norms;

    RefVectors = NormaliseVectors(RefVectors);
    cosines = normalisedObj*RefVectors';
    cosines = max(min(cosines,1),-1);
    angles = acos(cosines);
    [~,association] = min(angles,[],2);
    association(zeroMask) = 1;
    angles = angles(sub2ind(size(angles),(1:size(angles,1))',association));
end

function RefVectors = NormaliseVectors(RefVectors)
% Normalise reference vectors to unit length.

    if isempty(RefVectors)
        return;
    end
    norms = sqrt(sum(RefVectors.^2,2));
    norms(norms == 0) = 1;
    RefVectors = RefVectors ./ norms;
end
