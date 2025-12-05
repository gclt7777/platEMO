function Population = DistributionOptimization(Problem,Population,PV,CXV)
% Distribution optimization

if nargin < 3
    CXV = [];
end

N              = length(Population);
OffDec         = Population(TournamentSelection(2,N,sum(Population.objs,2))).decs;
ParentDec      = Population(randi(N,1,2*N)).decs;
NewDec         = OperatorGA(Problem,ParentDec);
if size(NewDec,1) > N
    NewDec = NewDec(1:N,:);
end
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
