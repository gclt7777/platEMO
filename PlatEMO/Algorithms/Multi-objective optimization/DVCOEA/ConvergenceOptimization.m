function [ Offspring ] = ConvergenceOptimization(Population,CVgroup )
% spopеCVgroup־߱Ż
[N,D] = size(Population.decs);
Con   = sum(Population.objs,2);
if isempty(CVgroup)
    Offspring = Population;
    return;
end
% Select parents
MatingPool = TournamentSelection(2,2*N,Con);
% Generate offsprings
OffDec = Population.decs;
rate = max(1,floor(D/length(CVgroup)/2));
NewDec = DE(Population.decs,Population(MatingPool(1:end/2)).decs,...
    Population(MatingPool(end/2+1:end)).decs,...
    {1,0.5,rate,20});

OffDec(:,CVgroup) = NewDec(:,CVgroup);
Offspring = INDIVIDUAL(OffDec);
end
