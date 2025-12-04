function DVCOEA(Global)
%% Parameter setting
[nSel,nPer,nCor] = Global.ParameterSet(5,50,5);
%% Generate random population
Archive = Global.Initialization();
%% 用于VariableCluster和CorrelationAnalysis
%% Detect the group of each convergence-related variables
% 决策变量分类
% CV:convergence-related variables;CO:contribute objectives of CV
% DV:diversity-related variables
[CV,DV,CO] = VariableClustering(Global,Archive,nSel,nPer);
% 相互作用分析后得到的收敛性决策变量分组
CVgroup = CorrelationAnalysis(Global,Archive,CV,nCor);
CXV = [];
for i = 1:length(CVgroup)
   if length(CVgroup{i}) > 1
       CXV = [CXV,CVgroup{i}];
   end
end
% 按贡献目标分组
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
    subPop = max(1,floor(length(Archive)/Global.M));
    for m = 1:Global.M
        start = (m-1)*subPop + 1;
        if m == Global.M
            range = start:length(Archive);
        else
            range = start:m*subPop;
        end
        if ~isempty(subSet{m})
            Archive(range) = ConvergenceOptimization(Archive(range),subSet{m});
        end
    end
    % Distribution optimization
    Archive = DistributionOptimization(Archive,DV,CXV);
end
end
