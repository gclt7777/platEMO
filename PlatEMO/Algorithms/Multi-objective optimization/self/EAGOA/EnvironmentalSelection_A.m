function Population = EnvironmentalSelection_A(Population,N)
% The environmental selection of distribution optimization in LMEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = FrontNo < MaxFNo;

    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    
    
    Next2 = 1 : length(Last);
    [Fitness,I,C] = CalFitness(Population(Last).objs,0.05);
    while length(Next2) > N-sum(Next)
        [~,x]   = min(Fitness(Next2));
        Fitness = Fitness + exp(-I(Next2(x),:)/C(Next2(x))/0.05);
        Next2(x) = [];
    end
    Next(Last(Next2)) = true;
    Population = Population(Next);
    
    

%         Next2 = 1 : length(Population);
%     [Fitness,I,C] = CalFitness(Population.objs,0.05);
%     while length(Next2) > N
%         [~,x]   = min(Fitness(Next2));
%         Fitness = Fitness + exp(-I(Next2(x),:)/C(Next2(x))/0.05);
%         Next2(x) = [];
%     end
%     Population = Population(Next2);

    
    
    
    
%     temp = Population.objs;
%     deltaS = CalHV(temp,max(temp,[],1)*1.1,1,10000);
%     [~,Choose] = sort(deltaS,'descend');
%     Choose = Choose(1:N);
%     Next = false(1,size(Population, 1));
%     Next(Choose) = true;


%     
%     temp = Population(Last).objs;
%     
%       Fitness = sum(temp.*temp,2);
% %      [~,Choose] = sort(Fitness,'descend');
%       [~,Choose] = sort(Fitness);
%       Choose = Choose(1:N-sum(Next));
%       Next(Last(Choose)) = true;
    
    
%     temp = Population(Last).decs * R';
%     [Fitness,~] = NDSort(temp,inf);
%     [~,Choose] = sort(Fitness);
%     Choose = Choose(1:N-sum(Next));
%     Next(Last(Choose)) = true;



%    Choose = Truncation(Population(Last).objs,N-sum(Next));
%    Next(Last(Choose)) = true;
%     
% 
%     
%     % Population for next generation
%     Population = Population(Next);
end

function Choose = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Calculate the normalized angle between each two solutions
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,size(PopObj,1),1))./repmat(fmax-fmin,size(PopObj,1),1);
    Cosine = 1 - pdist2(PopObj,PopObj,'cosine');
    Cosine(logical(eye(length(Cosine)))) = 0;
    
    %% Truncation
    % Choose the extreme solutions first
    Choose = false(1,size(PopObj,1)); 
    [~,extreme] = max(PopObj,[],1);
    Choose(extreme) = true;
    % Choose the rest by truncation
    if sum(Choose) > K
        selected = find(Choose);
        Choose   = selected(randperm(length(selected),K));
    else
        while sum(Choose) < K
            unSelected = find(~Choose);
            [~,x]      = min(max(Cosine(~Choose,Choose),[],2));
            Choose(unSelected(x)) = true;
        end
    end
end

function Fitness = calFitness(PopObj)
% Calculate the fitness by shift-based density

    N      = size(PopObj,1);
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
    Dis    = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    Fitness = min(Dis,[],2);
end

function [Fitness,I,C] = CalFitness(PopObj,kappa)

    N = size(PopObj,1);
    PopObj = (PopObj-repmat(min(PopObj),N,1))./(repmat(max(PopObj)-min(PopObj),N,1));
    I      = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(PopObj(i,:)-PopObj(j,:));
        end
    end
    C = max(abs(I));
    Fitness = sum(-exp(-I./repmat(C,N,1)/kappa)) + 1;
end



