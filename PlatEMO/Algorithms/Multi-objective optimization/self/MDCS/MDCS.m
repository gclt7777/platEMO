function MDCS(Global)
% <algorithm> <L>
% Multi-directional collaborative search (MDCS).
%
% This implementation follows the original algorithmic structure while
% adapting it to the function-handle based PlatEMO interface used in this
% repository.
%
%------------------------------- Reference --------------------------------
% X. Li, B. Xin, H. Gao, J. Zhang and W. Hao, Multi-directional
% collaborative search for many-objective optimization problems,
% Information Sciences, 2022, 607: 1235-1255.
%--------------------------------------------------------------------------

    %% Parameter setting
    [W,Global.N] = UniformPoint(Global.N,Global.M);
    [CAsize]     = Global.ParameterSet(Global.N);

    %% Generate random population and initialize the two archives
    Population = Global.Initialization();
    [CA,Fit]   = UpdateCA([],Population,CAsize);
    DA         = UpdateDA([],Population,Global.N,W);

    %% Optimization
    while Global.NotTermination(DA)
        FE    = Global.evaluated;
        maxFE = Global.evaluation;
        [ParentC,ParentM] = MatingSelection(CA,DA,Fit,FE,maxFE,Global.N);

        Offspring = [GenerateOffspring(ParentC,{1,15,0,0}), ...
                     GenerateOffspring(ParentM,{0,0,1,15})];
        if isempty(Offspring)
            continue;
        end

        [CA,Fit] = UpdateCA(CA,Offspring,CAsize);
        DA       = UpdateDA(DA,Offspring,Global.N,W);
    end
end

function Offspring = GenerateOffspring(Parents,Parameter)
% Ensure a valid set of parents for GA and generate offspring.

    if isempty(Parents)
        Offspring = INDIVIDUAL.empty();
        return;
    end

    numParents = numel(Parents);
    if numParents == 1
        Parents(2) = Parents(1);
    elseif mod(numParents,2) == 1
        Parents(end+1) = Parents(randi(numParents));
    end

    Offspring = GA(Parents,Parameter);
end
