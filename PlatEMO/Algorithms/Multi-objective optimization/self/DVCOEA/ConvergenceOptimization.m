function Population = ConvergenceOptimization(Problem,Population,CVgroup)
% Convergence optimization for a group of convergence-related variables

    PopDec = Population.decs;
    N      = size(PopDec,1);
    D      = size(PopDec,2);
    CVgroup = reshape(CVgroup,1,[]);
    CVgroup = CVgroup(~isnan(CVgroup));
    CVgroup = unique(round(CVgroup),'stable');
    CVgroup = CVgroup(CVgroup>=1 & CVgroup<=D);
    if isempty(CVgroup)
        return;
    end
    % Select parents
    Con         = sum(Population.objs,2);
    MatingPool  = TournamentSelection(2,2*N,Con);
    rate        = max(1,floor(D/length(CVgroup)/2));
    OffDec      = PopDec;
    NewDec      = OperatorDE(Problem,PopDec,...
        Population(MatingPool(1:end/2)).decs,...
        Population(MatingPool(end/2+1:end)).decs,...
        {1,0.5,rate,20});
    OffDec(:,CVgroup) = NewDec(:,CVgroup);
    Offspring         = Problem.Evaluation(OffDec);
    better            = all(Offspring.objs<=Population.objs,2) & any(Offspring.objs<Population.objs,2);
    Population(better) = Offspring(better);
end