function DVCOEA(Global)
%% Parameter setting
[nSel,nPer,nCor] = Global.ParameterSet(5,50,5);
%% Generate random population
Archive = Global.Initialization();
%% VariableClusterCorrelationAnalysis
%% Detect the group of each convergence-related variables
% ߱
% CV:convergence-related variables;CO:contribute objectives of CV
% DV:diversity-related variables
[CV,DV,CO] = VariableClustering(Global,Archive,nSel,nPer);
% ໥÷õԾ߱
CVgroup = CorrelationAnalysis(Global,Archive,CV,nCor);
CXV = [];
for i = 1:length(CVgroup)
   if length(CVgroup{i}) > 1
       CXV = [CXV,CVgroup{i}];
   end
end
% Ŀ
subSet = cell(1,Global.M);
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
while Global.NotTermination(Archive)
    % Convergence optimization
    subPop = ceil(length(Archive)/Global.M);
    for m = 1:Global.M
        index = (m-1)*subPop+1:min(m*subPop,length(Archive));
        if isempty(index)
            continue;
        end
        Archive(index) = ConvergenceOptimization(Archive(index),subSet{m});
    end
    % Distribution optimization
    Archive = DistributionOptimization(Archive,DV,CXV);
end
end