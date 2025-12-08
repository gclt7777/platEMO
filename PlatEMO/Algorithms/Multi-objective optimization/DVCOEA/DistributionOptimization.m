function Population = DistributionOptimization(Population,DV,CXV)
% Distribution optimization

    if nargin < 3
        CXV = [];
    end
    PV = unique([DV,CXV]);

    N            = length(Population);
OffDec       = Population(TournamentSelection(2,N,sum(Population.objs,2))).decs;
NewDec       = GA(Population(randi(N,1,N)).decs);
OffDec(:,PV) = NewDec(:,PV);
Offspring    = INDIVIDUAL(OffDec);
Population   = EnvironmentalSelection([Population,Offspring],N);
end

