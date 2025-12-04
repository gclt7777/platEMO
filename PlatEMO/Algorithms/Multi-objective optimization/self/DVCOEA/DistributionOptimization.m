function Population = DistributionOptimization(Problem,Population,DV,CXV)
% Distribution optimization focusing on diversity-related variables

    if nargin < 3
        CXV = [];
    end

    PopDec = Population.decs;
    D      = size(PopDec,2);

    DV  = reshape(DV,1,[]);
    CXV = reshape(CXV,1,[]);
    PV  = [DV(~isnan(DV)),CXV(~isnan(CXV))];
    PV  = unique(round(PV),'stable');
    PV  = PV(PV>=1 & PV<=D);

    if isempty(PV)
        return;
    end

    N        = length(Population);
    Fitness  = sum(Population.objs,2);
    Parents  = Population(TournamentSelection(2,N,Fitness));
    OffDec   = Parents.decs;

    parentCount = max(2,2*ceil(N/2));
    parentIdx   = randi(N,1,parentCount);
    parentDec   = Population(parentIdx).decs;
    gaOffspring = OperatorGA(Problem,parentDec);
    if size(gaOffspring,1) < N
        repFactor   = ceil(N/size(gaOffspring,1));
        gaOffspring = repmat(gaOffspring,repFactor,1);
    end
    NewDec = gaOffspring(1:N,:);

    maxCols = min([size(OffDec,2),size(NewDec,2),D]);
    PV      = PV(PV<=maxCols);
    if isempty(PV)
        return;
    end

    OffDec(:,PV) = NewDec(:,PV);
    Offspring    = Problem.Evaluation(OffDec);
    Population   = EnvironmentalSelection([Population,Offspring],N);
end
