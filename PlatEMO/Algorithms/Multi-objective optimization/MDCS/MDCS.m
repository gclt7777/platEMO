classdef MDCS < ALGORITHM
% <2022> <many> <real/integer/binary/permutation/label> <multimodal>
% Multi-directional collaborative search (MDCS).
%
%------------------------------- Reference --------------------------------
% X. Li, B. Xin, H. Gao, J. Zhang and W. Hao, Multi-directional collaborative
% search for many-objective optimization problems, Information Sciences,
% 2022, 607: 1235-1255.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            CAsize        = Algorithm.ParameterSet(Problem.N);
            CAsize        = max(1,min(Problem.N,round(CAsize)));

            %% Generate random population and initialize the two archives
            Population = Problem.Initialization();
            [CA,Fit]   = UpdateCA(SOLUTION.empty(),Population,CAsize);
            DA         = UpdateDA(SOLUTION.empty(),Population,Problem.N,W);

            %% Optimization
            while Algorithm.NotTerminated(DA)
                FE    = Problem.FE;
                maxFE = Problem.maxFE;
                [ParentC,ParentM] = MatingSelection(CA,DA,Fit,FE,maxFE,Problem.N);

                Offspring = [MDCS_Generate(Problem,ParentC,{1,15,0,0}), ...
                             MDCS_Generate(Problem,ParentM,{0,0,1,15})];
                if isempty(Offspring)
                    continue;
                end

                [CA,Fit] = UpdateCA(CA,Offspring,CAsize);
                DA       = UpdateDA(DA,Offspring,Problem.N,W);
            end
        end
    end
end

function Offspring = MDCS_Generate(Problem,Parents,Parameter)
% Ensure a valid set of parents for GA and generate offspring.

    if isempty(Parents)
        Offspring = SOLUTION.empty();
        return;
    end

    numParents = numel(Parents);
    if numParents == 1
        Parents(2) = Parents(1);
    elseif mod(numParents,2) == 1
        Parents(end+1) = Parents(randi(numParents));
    end

    if exist('OperatorGA','file') == 2
        Offspring = OperatorGA(Problem,Parents,Parameter);
    else
        Offspring = GA(Parents,Parameter);
    end
end
