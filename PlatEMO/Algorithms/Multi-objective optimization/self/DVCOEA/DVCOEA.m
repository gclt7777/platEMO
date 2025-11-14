classdef DVCOEA < ALGORITHM
% <2023> <multi> <real/integer> <large/none>
% Decomposition-based variable clustering optimization evolutionary algorithm

%------------------------------- Reference --------------------------------
% J. Wang, Y. Sun, Y. Jin, and X. Yao. Decomposition-based co-evolutionary
% algorithm for large-scale multiobjective optimization. IEEE Transactions
% on Evolutionary Computation, 2019, 23(2): 232-246.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [nSel,nPer,nCor] = Algorithm.ParameterSet(5,50,5);

            %% Generate random population
            Population = Problem.Initialization();

            %% Variable clustering and correlation analysis
            [CV,DV,CO] = VariableClustering(Problem,Population,nSel,nPer);
            CVgroup    = CorrelationAnalysis(Problem,Population,CV,nCor);
            CXV        = [];
            for i = 1 : length(CVgroup)
                if numel(CVgroup{i}) > 1
                    CXV = [CXV,CVgroup{i}]; %#ok<AGROW>
                end
            end
            subSet = cell(1,Problem.M);
            for i = 1 : length(CV)
                conum = length(CO{CV(i)});
                if conum == 1
                    m = CO{CV(i)};
                else
                    m = CO{CV(i)}(randi(conum));
                end
                subSet{m} = [subSet{m},CV(i)]; %#ok<AGROW>
            end

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Convergence optimization
                for m = 1 : Problem.M
                    if ~isempty(subSet{m})
                        Population = ConvergenceOptimization(Problem,Population,subSet{m});
                    end
                end
                % Distribution optimization
                Population = DistributionOptimization(Problem,Population,DV,CXV);
            end
        end
    end
end
