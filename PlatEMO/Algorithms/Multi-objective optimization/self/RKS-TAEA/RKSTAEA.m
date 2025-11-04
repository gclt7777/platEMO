function RKSTAEA(Global)
% <algorithm> <L>
%RKSTAEA - Reference vector assisted knee search with two-archive strategy.
%
%   This function-based entry point keeps the original implementation logic
%   while remaining compatible with legacy (function-handle) and the newer
%   class-oriented PlatEMO dispatching styles that still invoke algorithms as
%   callable handles. No algorithmic behaviour has been altered – the only
%   adjustments ensure that the required data (population size, operators and
%   termination checks) are obtained through the provided GLOBAL interface.

    %% Parameter setting
    CAsize = Global.ParameterSet(Global.N);
    CAsize = max(1,min(Global.N,round(CAsize)));

    %% Generate random population
    Population = Global.Initialization();
    [DA,p] = UpdateDA([],Population,Global.N);
    CA = UpdateCA([],Population,CAsize,p);

    %% Optimization
    while Global.NotTermination(DA)
        [~,SDEP_fitness,~] = SDE_plus_indicator(CA.objs,1,p);
        MatingPool = TournamentSelection(2,Global.N,-SDEP_fitness);
        Offspring  = RKSTAEA_Generate(Global,CA(MatingPool));
        [DA,p] = UpdateDA(DA,Offspring,Global.N);
        CA = UpdateCA(CA,Offspring,CAsize,p);
    end
end

function Offspring = RKSTAEA_Generate(Global,Parents)
%RKSTAEA_GENERATE Generate offspring using the available GA operator.
%
%   The original RKS-TAEA relies on PlatEMO's GA operator. Recent PlatEMO
%   releases expose it as OperatorGA(PROBLEM,Parents) while legacy versions
%   provide GA(Parents). To keep strict behavioural equivalence we call the
%   most specific operator that exists on the MATLAB path and fall back to
%   the classic GA when OperatorGA is unavailable.

    if exist('OperatorGA','file') == 2
        Offspring = OperatorGA(Global.problem,Parents);
    else
        Offspring = GA(Parents);
    end
end
