function Population = DistributionOptimization(Problem,Population,PV,CXV)
% Distribution optimization

if nargin < 3
    CXV = [];
end

N              = length(Population);
OffDec         = Population(TournamentSelection(2,N,sum(Population.objs,2))).decs;
NewDec         = GA(Population(randi(N,1,N)).decs);
maxDecColumns  = min(size(OffDec,2),size(NewDec,2));
PV             = unique([PV,CXV]);
PV             = PV(PV >= 1 & PV <= maxDecColumns & isfinite(PV));

if isempty(PV) || maxDecColumns == 0
    return;
end

OffDec(:,PV)   = NewDec(:,PV);
Offspring      = Problem.Evaluation(OffDec);
Population     = EnvironmentalSelection([Population,Offspring],N);
end
