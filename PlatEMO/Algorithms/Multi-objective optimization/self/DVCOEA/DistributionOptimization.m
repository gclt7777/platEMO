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
    ParentDec = Population(randi(N,1,N)).decs;
    if mod(size(ParentDec,1),2) == 1
        ParentDec = ParentDec([1:end,1],:);
    end
    NewDec  = OperatorGA(Problem,ParentDec);
    OffDec(:,DV) = NewDec(1:N,DV);
    Offspring    = Problem.Evaluation(OffDec);
    Population   = EnvironmentalSelection([Population,Offspring],N);
end
