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
    parentDec = Population(randi(N,1,N)).decs;
    NewDec   = GA(parentDec);

    maxCols = min([size(OffDec,2),size(NewDec,2),D]);
    PV      = PV(PV<=maxCols);
    if isempty(PV)
        return;
    end

    OffDec(:,PV) = NewDec(:,PV);
    Offspring    = Problem.Evaluation(OffDec);
    Population   = EnvironmentalSelection([Population,Offspring],N);
end