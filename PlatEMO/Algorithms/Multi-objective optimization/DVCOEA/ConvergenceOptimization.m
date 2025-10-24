function Population = ConvergenceOptimization(Problem,Population,CVgroup)
% Convergence optimization for a group of convergence-related variables

    if isempty(CVgroup)
        return;
    end
    N = length(Population);
    D = size(Population.decs,2);
    % Select parents
    Con         = sum(Population.objs,2);
    MatingPool  = TournamentSelection(2,2*N,Con);
    OffDec      = Population.decs;
    NewDec      = OperatorDE(Problem,Population.decs,...
        Population(MatingPool(1:end/2)).decs,...
        Population(MatingPool(end/2+1:end)).decs,...
        {1,0.5,D/length(CVgroup)/2,20});
    OffDec(:,CVgroup) = NewDec(:,CVgroup);
    Offspring         = Problem.Evaluation(OffDec);
    better            = all(Offspring.objs<=Population.objs,2) & any(Offspring.objs<Population.objs,2);
    Population(better) = Offspring(better);
end
