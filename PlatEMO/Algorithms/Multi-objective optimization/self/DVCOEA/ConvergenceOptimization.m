function [ Offspring ] = ConvergenceOptimization(Problem,Population,CVgroup)
% 对spop中的CVgroup部分决策变量做优化
[N,D] = size(Population.decs);
if isempty(CVgroup)
    Offspring = Population;
    return;
end
CVgroup = CVgroup(CVgroup<=D);
if isempty(CVgroup)
    Offspring = Population;
    return;
end
Con   = sum(Population.objs,2);
% Select parents
MatingPool = TournamentSelection(2,2*N,Con);
% Generate offsprings
OffDec = Population.decs;
NewDec = DE(Population.decs,Population(MatingPool(1:end/2)).decs,...
    Population(MatingPool(end/2+1:end)).decs,...
    {1,0.5,max(1,D/length(CVgroup)/2),20});

OffDec(:,CVgroup) = NewDec(:,CVgroup);
Offspring = Problem.Evaluation(OffDec);
end
