classdef RKSTAEA < ALGORITHM
% <2019> <multi/many> <real/integer/binary/permutation/label> <none>
% Reference vector assisted knee search with two-archive strategy
%
%------------------------------- Reference --------------------------------
% Z. Liang, H. Li, J. Zheng, K. Li, and X. Yao. A knee point-driven evolutionary
% algorithm for many-objective optimization. IEEE Transactions on Evolutionary
% Computation, 2019, 23(3): 414-425.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            CAsize = Algorithm.ParameterSet(Problem.N);
            CAsize = max(1,min(Problem.N,round(CAsize)));

            %% Generate random population
            Population = Problem.Initialization();
            [DA,p] = UpdateDA([],Population,Problem.N);
            CA = UpdateCA([],Population,CAsize,p);

            %% Optimization
            while Algorithm.NotTerminated(DA)
                [~,SDEP_fitness,~] = SDE_plus_indicator(CA.objs,1,p);
                MatingPool = TournamentSelection(2,Problem.N,-SDEP_fitness);
                Offspring  = RKSTAEA_Generate(Problem,CA(MatingPool));
                [DA,p] = UpdateDA(DA,Offspring,Problem.N);
                CA = UpdateCA(CA,Offspring,CAsize,p);
            end
        end
    end
end

function Offspring = RKSTAEA_Generate(Problem,Parents)
%RKSTAEA_GENERATE Generate offspring using the available GA operator.
%
%   The original RKS-TAEA relies on PlatEMO's GA operator. Recent PlatEMO
%   releases expose it as OperatorGA(PROBLEM,Parents) while legacy versions
%   provide GA(Parents). To keep strict behavioural equivalence we call the
%   most specific operator that exists on the MATLAB path and fall back to
%   the classic GA when OperatorGA is unavailable.

    if exist('OperatorGA','file') == 2
        Offspring = OperatorGA(Problem,Parents);
    else
        Offspring = GA(Parents);
    end
end
