function Population = DistributionOptimization(Problem,Population,DV,CXV)
% Distribution optimization

    if nargin < 3
        CXV = [];
    end
    PV = unique([DV,CXV]);
    PV = PV(PV>=1 & PV<=Problem.D);
    if isempty(PV)
        return;
    end

    N            = length(Population);
    OffDec       = Population(TournamentSelection(2,N,sum(Population.objs,2))).decs;
    parentDec    = Population(randi(N,1,2*ceil(N/2))).decs;
    NewDec       = OperatorGA(Problem,parentDec,{1,20,1,20});
    NewDec       = NewDec(1:N,:);
    OffDec(:,PV) = NewDec(:,PV);
    Offspring    = Problem.Evaluation(OffDec);
    Population   = EnvironmentalSelection([Population,Offspring],N);
end

