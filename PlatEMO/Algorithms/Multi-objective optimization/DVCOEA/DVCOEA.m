classdef DVCOEA < ALGORITHM
% <2023> <multi/many> <real/integer/label/binary/permutation> <large>

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [nSel,nPer,nCor] = Algorithm.ParameterSet(5,50,5);

            %% Generate random population
            Archive = Problem.Initialization();

            %% VariableClusterCorrelationAnalysis
            %% Detect the group of each convergence-related variables
            % ځ
            % CV:convergence-related variables;CO:contribute objectives of CV
            % DV:diversity-related variables
            [CV,DV,CO] = VariableClustering(Problem,Archive,nSel,nPer);
            % ༥÷õդځ
            CVgroup = CorrelationAnalysis(Problem,Archive,CV,nCor);
            CXV = [];
            for i = 1:length(CVgroup)
                if length(CVgroup{i}) > 1
                    CXV = [CXV,CVgroup{i}];
                end
            end
            % Ŀ
            subSet = cell(1,Problem.M);
            for i = 1:length(CV)
                %   if
                conum = length(CO{CV(i)});
                if conum == 1
                    m = CO{CV(i)};
                    subSet{m} = [subSet{m},CV(i)];
                else
                    m = CO{CV(i)}(randi(conum));
                    subSet{m} = [subSet{m},CV(i)];
                end
            end

            %% Optimization
            while Algorithm.NotTerminated(Archive)
                % Convergence optimization
                subPopSize = ceil(length(Archive)/Problem.M);
                for m = 1:Problem.M
                    startIdx = (m-1)*subPopSize + 1;
                    if startIdx > length(Archive)
                        break;
                    end
                    endIdx = min(m*subPopSize,length(Archive));
                    Archive(startIdx:endIdx) = ConvergenceOptimization(Problem,Archive(startIdx:endIdx),subSet{m});
                end
                % Distribution optimization
                Archive = DistributionOptimization(Problem,Archive,DV,CXV);
            end
        end
    end
end
