function DVCOEA(Global)
% <algorithm> <L>
% Decomposition-based variable clustering optimization evolutionary algorithm
% nSel ---  5 --- Number of selected solutions for decision variable clustering
% nPer --- 50 --- Number of perturbations on each solution for decision variable clustering
% nCor ---  5 --- Number of selected solutions for decision variable interaction analysis

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

    %% Parameter setting
    [nSel,nPer,nCor] = Global.ParameterSet(5,50,5);

    %% Generate random population
    Population = Global.Initialization();

    %% Variable clustering and correlation analysis
    [CV,DV,CO] = VariableClustering(Global,Population,nSel,nPer);
    CVgroup    = CorrelationAnalysis(Global,Population,CV,nCor);
    CXV        = [];
    for i = 1 : length(CVgroup)
        if numel(CVgroup{i}) > 1
            CXV = [CXV,CVgroup{i}]; %#ok<AGROW>
        end
    end
    subSet = cell(1,Global.M);
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
    while Global.NotTermination(Population)
        % Convergence optimization
        for m = 1 : Global.M
            if ~isempty(subSet{m})
                Population = ConvergenceOptimization(Population,subSet{m});
            end
        end
        % Distribution optimization
        Population = DistributionOptimization(Population,DV,CXV);
    end
end