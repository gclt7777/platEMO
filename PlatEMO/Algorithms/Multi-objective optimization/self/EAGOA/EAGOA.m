classdef EAGOA < ALGORITHM
% <algorithm> <L>

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            [nSel,nPer,nCor,type] = Algorithm.ParameterSet(5,50,5,1); %#ok<ASGLU>

            Population = Problem.Initialization();
            D          = size(Population.decs,2);
            R          = Population.objs\Population.decs;
            [m1,m2]    = VariableClustering_A(Problem,Population,nSel,nPer);

            if isempty(m1)
                m1 = randi(D,1,max(1,floor(D/50)));
            end

            while Algorithm.NotTerminated(Population)
                if Problem.FE/Problem.maxFE < 0.9
                    temp = 5;
                else
                    temp = 1;
                end

                for j = 1 : temp*2 %#ok<FXSET>
                    for ij = 1 : length(m2)
                        drawnow();
                        Population = SingleOptimization(Problem,Population,m2(ij),R);
                    end
                end

                for j = 1 : temp %#ok<FXSET>
                    drawnow();
                    Population = GroupOptimization(Problem,Population,m1,R);
                end
            end
        end
    end
end

function P = GAhalf_PF(obj,N)
    P = [0:1/(N-1):1;1:-1/(N-1):0]';
    P = P./repmat(sqrt(sum(P.^2,2)),1,size(P,2));
    P = [P(:,ones(1,obj.Global.M-2)),P];
    P = P./sqrt(2).^repmat([obj.Global.M-2,obj.Global.M-2:-1:0],size(P,1),1);
end

function Con = calCon(PopuObj)
% Calculate the convergence of each solution

    FrontNo = NDSort(PopuObj,inf);
    Con     = sum(PopuObj,2);
    Con     = FrontNo'*(max(Con)-min(Con)) + Con;
end

function Offspring = GAhalf2(Parent, N)
    Offspring = Parent - Parent.*(rand(size(Parent))*2 - 1)*0.5;
end

function output = sigmoid(x)
    output =1./(1+exp(-x));
end

function output = swish(x)
    output =x./(1+exp(-x));
end

function output = ELU(x)
    output = 0.1*(exp(x)-1);
end

function output = FL(x)
    if x < 0
        output = x;
    else
        output = 0.5*x;
    end
end

function Offspring = GAhalf4(Parent, N)
    Offspring = Parent - Parent.*(rand(size(Parent))-0.5);
end

function Offspring = GAhalf3(Parent,N1)
    [proC,disC,proM,disM] = deal(1,20,1,20);

    Parent1 = Parent(1:floor(end/2),:);
    Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
    [N,D]   = size(Parent1);

    %% Genetic operators for real encoding
    % Simulated binary crossover
    beta = zeros(N,D);
    mu   = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2,
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
    % Polynomial mutation
    lower = min(Parent, [], 1)*0.01;
    upper = max(Parent, [], 1)*10;
    Lower = repmat(lower,2*N,1);
    Upper = repmat(upper,2*N,1);
    Site  = rand(2*N,D) < proM/D;
    mu    = rand(2*N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                          (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                          (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
end

function Offspring = GAhalf2_2(Parent, N)
    Offspring = Parent - Parent.*(rand(size(Parent)))*0.25;
end

function Offspring = GAhalf1(Parent, N)
    Offspring = Parent + Parent.*(rand(size(Parent)))*0.25;
end

function Offspring = GAhalf5(Parent, N)
    fmin     = 1.5*min(Parent,[],1) - 0.5*max(Parent,[],1);
    fmax     = 1.5*max(Parent,[],1) - 0.5*min(Parent,[],1);
    Offspring = unifrnd(repmat(fmin,N,1),repmat(fmax,N,1));
end

function Population2 = GroupOptimization(Problem,Population,Div_V,R)
    N      = length(Population);
    OffDec = Population(TournamentSelection(2,N,calCon(Population.objs))).decs;
    NewObjs  = GAhalf3(Population.objs, N);
    Offspring_Convergence  = ELU(NewObjs * R(:, Div_V));
    NewObjs2 = GAhalf2_2(Population.objs, N);
    Offspring_Convergence2 = ELU(NewObjs2 * R(:, Div_V));
    temp  = sqrt(OffDec(randperm(N),Div_V).*OffDec(randperm(N),Div_V));
    temp2 = OffDec(:,Div_V);
    a     = randperm(N);
    temp(temp>OffDec(a,Div_V)) = temp2(temp>OffDec(a,Div_V));
    OffDec(:,Div_V) = temp + (Offspring_Convergence-Offspring_Convergence2);
    OffDec(:,Div_V) = min(max(OffDec(:,Div_V),repmat(Problem.lower(Div_V),N,1)),repmat(Problem.upper(Div_V),N,1));
    Offspring      = Problem.Evaluation(OffDec);
    Population2    = EnvironmentalSelection_A([Population,Offspring],N);
end

function Population = SingleOptimization(Problem,Population,con_V,R)
    [N,~]  = size(Population.decs);
    OffDec = Population.decs;
    NewObjs  = GAhalf3(Population(randperm(N)).objs, N);
    Offspring_Convergence  = FL(NewObjs * R(:, con_V));
    NewObjs2 = GAhalf2_2(Population(randperm(N)).objs, N);
    Offspring_Convergence2 = FL(NewObjs2 * R(:, con_V));
    OffDec(:,con_V) = OffDec(randperm(N),con_V) + (Offspring_Convergence - Offspring_Convergence2);
    OffDec(:,con_V) = min(max(OffDec(:,con_V),repmat(Problem.lower(con_V),N,1)),repmat(Problem.upper(con_V),N,1));
    Offspring       = Problem.Evaluation(OffDec);
    updated         = sum(Offspring.objs <= Population.objs,2) >= Problem.M;
    Population(updated) = Offspring(updated);
end
