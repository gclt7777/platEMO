classdef DVCOEA < ALGORITHM
% <2023> <multi> <real/integer> <large/none>

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [nSel,nPer,nCor] = Algorithm.ParameterSet(5,50,5);
            %% Generate random population
            Archive = Problem.Initialization();
            %% 用于VariableCluster和CorrelationAnalysis
            %% Detect the group of each convergence-related variables
            % 决策变量分类
            % CV:convergence-related variables;CO:contribute objectives of CV
            % DV:diversity-related variables
            [CV,DV,CO] = VariableClustering(Problem,Archive,nSel,nPer);
            % 相互作用分析后得到的收敛性决策变量分组
            CVgroup = CorrelationAnalysis(Problem,Archive,CV,nCor);
            CXV = [];
            for i = 1:length(CVgroup)
                if length(CVgroup{i}) > 1
                    CXV = [CXV,CVgroup{i}];
                end
            end
            % 按贡献目标分组
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
                subPop = max(1,floor(length(Archive)/Problem.M));
                for m = 1:Problem.M
                    start = (m-1)*subPop + 1;
                    if m == Problem.M
                        range = start:length(Archive);
                    else
                        range = start:m*subPop;
                    end
                    if ~isempty(subSet{m})
                        Archive(range) = ConvergenceOptimization(Problem,Archive(range),subSet{m});
                    end
                end
                % Distribution optimization
                Archive = DistributionOptimization(Problem,Archive,DV,CXV);
            end
        end
    end
end
