function [ Offspring ] = ConvergenceOptimization(Population,CVgroup )
% 对spop中的CVgroup部分决策变量做优化
[N,D] = size(Population.decs);
Con   = sum(Population.objs);
% Select parents
MatingPool = TournamentSelection(2,2*N,Con);
% Generate offsprings
OffDec = Population.decs;
NewDec = DE(Population.decs,Population(MatingPool(1:end/2)).decs,...
    Population(MatingPool(end/2+1:end)).decs,...
    {1,0.5,D/length(CVgroup)/2,20});

OffDec(:,CVgroup) = NewDec(:,CVgroup);
Offspring = INDIVIDUAL(OffDec);
end
