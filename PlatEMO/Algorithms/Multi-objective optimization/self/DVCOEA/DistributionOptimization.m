function Population = DistributionOptimization(Population,DV,CXV)
% Distribution optimization focusing on diversity-related variables

    if nargin < 3
        CXV = [];
    end
    DV = setdiff(DV,unique(CXV));
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
    Offspring    = INDIVIDUAL(OffDec);
    Population   = EnvironmentalSelection([Population,Offspring],N);
end