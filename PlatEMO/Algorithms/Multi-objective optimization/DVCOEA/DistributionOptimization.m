function Population = DistributionOptimization(Population,DV,CXV)
% Distribution optimization

N            = length(Population);
PV           = unique([DV(:);CXV(:)])';
OffDec       = Population(TournamentSelection(2,N,sum(Population.objs,2))).decs;
NewDec       = GA(Population(randi(N,1,N)).decs);
if ~isempty(PV)
    OffDec(:,PV) = NewDec(:,PV);
end
Offspring    = INDIVIDUAL(OffDec);
Population   = EnvironmentalSelection([Population,Offspring],N);
end

