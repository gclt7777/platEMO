function Population = DistributionOptimization(Population,PV)
% Distribution optimization

N            = length(Population);
OffDec       = Population(TournamentSelection(2,N,sum(Population.objs,2))).decs;
NewDec       = GA(Population(randi(N,1,N)).decs);
OffDec(:,PV) = NewDec(:,PV);
Offspring    = INDIVIDUAL(OffDec);
Population   = EnvironmentalSelection([Population,Offspring],N);
end

