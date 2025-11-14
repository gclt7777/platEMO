function Population = DistributionOptimization(Problem,Population,DV,CXV)
% Distribution optimization focusing on diversity-related variables

    if nargin < 3
        CXV = [];
    end

    PopDec = Population.decs;
    D      = size(PopDec,2);

    DV  = reshape(DV,1,[]);
    CXV = reshape(CXV,1,[]);
    DV  = DV(~isnan(DV));
    CXV = CXV(~isnan(CXV));
    DV  = unique(round(DV),'stable');
    CXV = unique(round(CXV),'stable');
    DV  = DV(~ismember(DV,CXV));
    DV  = DV(DV>=1 & DV<=D);
    if isempty(DV)
        return;
    end

    N       = length(Population);
    Fitness = sum(Population.objs,2);
    Parents = Population(TournamentSelection(2,N,Fitness));
    OffDec  = Parents.decs;
    parentDec = Population(randi(N,1,N)).decs;
    NewDec  = GA(parentDec);
    OffDec(:,DV) = NewDec(:,DV);
    Offspring    = Problem.Evaluation(OffDec);
    Population   = EnvironmentalSelection([Population,Offspring],N);
end