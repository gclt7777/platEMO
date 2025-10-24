function Population = DistributionOptimization(Problem,Population,DV,CXV)
% Distribution optimization focusing on diversity-related variables

    if nargin < 4
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
    parentDec = Population(randi(N,1,2*ceil(N/2))).decs;
    NewDec  = OperatorGA(Problem,parentDec);
    NewDec  = NewDec(1:size(OffDec,1),:);
    OffDec(:,DV) = NewDec(:,DV);
    Offspring    = Problem.Evaluation(OffDec);
    Population   = EnvironmentalSelection([Population,Offspring],N);
end
