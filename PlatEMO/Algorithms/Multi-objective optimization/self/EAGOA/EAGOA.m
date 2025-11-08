classdef EAGOA < ALGORITHM
    % <2023> <multi/many> <real/integer/label/binary/permutation> <large>
    % Enhanced Objective Space-based Adaptive Optimization (EAGOA)

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [nSel,nPer,~,~] = Algorithm.ParameterSet(5,50,5,1);

            %% Generate random population
            Population = Problem.Initialization();
            N = length(Population);
            D = size(Population.decs,2);

            R = Population.objs\Population.decs;
            [m1,m2] = EAGOA.VariableClustering(Problem,Population,nSel,nPer);
            if isempty(m1)
                count = max(1,ceil(D/50));
                m1    = uint16(rand(1,count)*(D-1)+1);
            end

            while Algorithm.NotTerminated(Population)
                if Problem.FE / Problem.maxFE < 0.9
                    temp = 5;
                else
                    temp = 1;
                end

                for j = 1 : temp*2
                    if Problem.FE >= Problem.maxFE
                        break;
                    end
                    for ij = 1 : length(m2)
                        if Problem.FE >= Problem.maxFE
                            break;
                        end
                        drawnow();
                        Population = EAGOA.SingleOptimization(Problem,Population,m2(ij),R);
                    end
                end

                for j = 1 : temp
                    if Problem.FE >= Problem.maxFE
                        break;
                    end
                    drawnow();
                    Population = EAGOA.GroupOptimization(Problem,Population,m1,R);
                end
            end
        end
    end

    methods(Static, Access = private)
        function [PV,DV] = VariableClustering(Problem,Population,nSel,nPer)
            [N,D] = size(Population.decs);
            ND    = NDSort(Population.objs,1) == 1;
            fmin  = min(Population(ND).objs,[],1);
            fmax  = max(Population(ND).objs,[],1);
            if any(fmax==fmin)
                fmax = ones(size(fmax));
                fmin = zeros(size(fmin));
            end

            Angle  = zeros(D,nSel);
            RMSE   = zeros(D,nSel);
            Sample = randi(N,1,nSel);
            for i = 1 : D
                drawnow();
                if Problem.maxFE - Problem.FE < nSel*nPer
                    break;
                end
                Decs      = repmat(Population(Sample).decs,nPer,1);
                Decs(:,i) = unifrnd(Problem.lower(i),Problem.upper(i),size(Decs,1),1);
                newPopu   = Problem.Evaluation(Decs);
                if length(newPopu) < nSel*nPer
                    break;
                end
                for j = 1 : nSel
                    Points = newPopu(j:nSel:end).objs;
                    Points = (Points-repmat(fmin,size(Points,1),1))./repmat(fmax-fmin,size(Points,1),1);
                    Points = Points - repmat(mean(Points,1),nPer,1);
                    [~,~,V] = svd(Points);
                    Vector  = V(:,1)'./norm(V(:,1)');
                    error = zeros(1,nPer);
                    for k = 1 : nPer
                        error(k) = norm(Points(k,:)-sum(Points(k,:).*Vector)*Vector);
                    end
                    RMSE(i,j) = sqrt(sum(error.^2));
                    normal     = ones(1,size(Vector,2));
                    sine       = abs(sum(Vector.*normal,2))./norm(Vector)./norm(normal);
                    Angle(i,j) = real(asin(sine)/pi*180);
                end
            end

            VariableKind = (mean(RMSE,2)<1e-2)';
            result       = kmeans(Angle,2)';
            if any(result(VariableKind)==1) && any(result(VariableKind)==2)
                if mean(mean(Angle(result==1&VariableKind,:))) > mean(mean(Angle(result==2&VariableKind,:)))
                    VariableKind = VariableKind & result==1;
                else
                    VariableKind = VariableKind & result==2;
                end
            end
            PV = find(~VariableKind);
            DV = find(VariableKind);
        end

        function Population2 = GroupOptimization(Problem,Population,Div_V,R)
            N = length(Population);
            if isempty(Div_V)
                Population2 = Population;
                return;
            end
            if Problem.FE >= Problem.maxFE
                Population2 = Population;
                return;
            end
            OffDec = Population(TournamentSelection(2,N,EAGOA.calCon(Population.objs))).decs;
            NewObjs  = EAGOA.GAhalf3(Population.objs,N);
            Offspring_Convergence  = EAGOA.ELU(NewObjs * R(:,Div_V));
            NewObjs2 = EAGOA.GAhalf2_2(Population.objs,N);
            Offspring_Convergence2 = EAGOA.ELU(NewObjs2 * R(:,Div_V));
            temp  = sqrt(OffDec(randperm(N),Div_V).*OffDec(randperm(N),Div_V));
            temp2 = OffDec(:,Div_V);
            a     = randperm(N);
            temp(temp>OffDec(a,Div_V)) = temp2(temp>OffDec(a,Div_V));
            OffDec(:,Div_V) = temp + (Offspring_Convergence-Offspring_Convergence2);
            OffDec(:,Div_V) = min(max(OffDec(:,Div_V),repmat(Problem.lower(Div_V),N,1)),repmat(Problem.upper(Div_V),N,1));

            [Offspring,~] = EAGOA.EvaluateWithBudget(Problem,OffDec);
            if isempty(Offspring)
                Population2 = Population;
                return;
            end
            Population2 = EnvironmentalSelection_A([Population,Offspring],N);
        end

        function Population = SingleOptimization(Problem,Population,con_V,R)
            [N,~] = size(Population.decs);
            if isempty(con_V)
                return;
            end
            if Problem.FE >= Problem.maxFE
                return;
            end
            OffDec = Population.decs;

            NewObjs  = EAGOA.GAhalf3(Population(randperm(N)).objs,N);
            Offspring_Convergence  = EAGOA.FL(NewObjs * R(:,con_V));
            NewObjs2 = EAGOA.GAhalf2_2(Population(randperm(N)).objs,N);
            Offspring_Convergence2 = EAGOA.FL(NewObjs2 * R(:,con_V));

            OffDec(:,con_V) = OffDec(randperm(N),con_V) + (Offspring_Convergence-Offspring_Convergence2);
            OffDec(:,con_V) = min(max(OffDec(:,con_V),repmat(Problem.lower(con_V),N,1)),repmat(Problem.upper(con_V),N,1));

            remain = Problem.maxFE - Problem.FE;
            if remain <= 0
                return;
            elseif remain < N
                evalIdx = randperm(N,remain);
            else
                evalIdx = 1:N;
            end
            [Offspring,evalIdx] = EAGOA.EvaluateWithBudget(Problem,OffDec,evalIdx);
            if isempty(evalIdx)
                return;
            end

            updated = sum(Offspring.objs <= Population(evalIdx).objs,2) >= Problem.M;
            Population(evalIdx(updated)) = Offspring(updated);
        end

        function Offspring = GAhalf2(Parent,~)
            Offspring = Parent - Parent.*(rand(size(Parent))*2 - 1)*0.5;
        end

        function output = sigmoid(x)
            output = 1./(1+exp(-x));
        end

        function output = swish(x)
            output = x./(1+exp(-x));
        end

        function output = ELU(x)
            output = 0.1*(exp(x)-1);
        end

        function output = FL(x)
            output = x;
            output(x>=0) = 0.5*output(x>=0);
        end

        function Offspring = GAhalf4(Parent,~)
            Offspring = Parent - Parent.*(rand(size(Parent))-0.5);
        end

        function Offspring = GAhalf3(Parent,~)
            [proC,disC,proM,disM] = deal(1,20,1,20);
            lower = min(Parent,[],1)*0.01;
            upper = max(Parent,[],1)*10;
            Parent1 = Parent(1:floor(end/2),:);
            Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
            [Nsize,D] = size(Parent1);
            beta = zeros(Nsize,D);
            mu   = rand(Nsize,D);
            beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
            beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
            beta = beta.*(-1).^randi([0,1],Nsize,D);
            beta(rand(Nsize,D)<0.5) = 1;
            beta(repmat(rand(Nsize,1)>proC,1,D)) = 1;
            Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2,
                         (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
            Lower = repmat(lower,2*Nsize,1);
            Upper = repmat(upper,2*Nsize,1);
            Site  = rand(2*Nsize,D) < proM/D;
            mu    = rand(2*Nsize,D);
            temp  = Site & mu<=0.5;
            Offspring       = min(max(Offspring,Lower),Upper);
            Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
            temp = Site & mu>0.5;
            Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
        end

        function Offspring = GAhalf2_2(Parent,~)
            Offspring = Parent - Parent.*(rand(size(Parent)))*0.25;
        end

        function Offspring = GAhalf1(Parent,~)
            Offspring = Parent + Parent.*(rand(size(Parent)))*0.25;
        end

        function Offspring = GAhalf5(Parent,N)
            fmin = 1.5*min(Parent,[],1) - 0.5*max(Parent,[],1);
            fmax = 1.5*max(Parent,[],1) - 0.5*min(Parent,[],1);
            Offspring = unifrnd(repmat(fmin,N,1),repmat(fmax,N,1));
        end

        function Con = calCon(PopuObj)
            FrontNo = NDSort(PopuObj,inf);
            Con     = sum(PopuObj,2);
            Con     = FrontNo'*(max(Con)-min(Con)) + Con;
        end

        function [Offspring,idx] = EvaluateWithBudget(Problem,Decs,idx)
            remain = Problem.maxFE - Problem.FE;
            if remain <= 0
                Offspring = [];
                idx       = [];
                return;
            end
            total = size(Decs,1);
            if nargin < 3
                if remain < total
                    idx = randperm(total,remain);
                else
                    idx = 1:total;
                end
            else
                idx = idx(:)';
                if isempty(idx)
                    Offspring = [];
                    idx       = [];
                    return;
                end
                if remain < length(idx)
                    idx = idx(1:remain);
                end
            end
            if isempty(idx)
                Offspring = [];
                return;
            end
            idx       = idx(:);
            Offspring = Problem.Evaluation(Decs(idx,:));
        end
    end
end
