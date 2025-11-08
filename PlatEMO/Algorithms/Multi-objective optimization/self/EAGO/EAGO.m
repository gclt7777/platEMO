classdef EAGO < ALGORITHM
    % <2023> <multi/many> <real/integer/label/binary/permutation> <large>
    % Objective Space-based Population Generation (EAGO)
    % Deng et al., IEEE Transactions on Evolutionary Computation, 2023

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [nSel,nPer,c_fmin,c_fmax] = Algorithm.ParameterSet(5,30,0.01,10);

            %% Generate random population
            Population = Problem.Initialization();
            objs = Population.objs;
            x1   = Population.decs;
            R    = objs\x1;
            R1   = R;

            [Div_V,con_V] = EAGO.VariableClustering(Problem,Population,nSel,nPer);
            [Div_V1,con_V1,con_V_plus] = EAGO.VariableAnalysis1(Problem,Population,nSel,nPer,R1);

            change     = true;
            changeState = struct('initialized',false,'temp',1,'conPass',0,'conTotal',0, ...
                                 'conIdx',0,'distPass',0,'distTotal',0);
            intensiveState = struct('initialized',false,'step',0);

            while Algorithm.NotTerminated(Population)
                if change
                    if ~changeState.initialized
                        if Problem.FE / Problem.maxFE < 0.9
                            changeState.temp = 10;
                        else
                            changeState.temp = 1;
                        end

                        if ~isempty(con_V)
                            changeState.conTotal = max(1,changeState.temp);
                        else
                            changeState.conTotal = 0;
                        end
                        if ~isempty(Div_V)
                            changeState.distTotal = max(1,changeState.temp);
                        else
                            changeState.distTotal = 0;
                        end

                        if changeState.conTotal==0 && changeState.distTotal==0
                            change = false;
                            changeState.initialized = false;
                            intensiveState.initialized = false;
                            continue;
                        end

                        changeState.conPass  = 0;
                        changeState.conIdx   = 0;
                        changeState.distPass = 0;
                        changeState.initialized = true;
                    end

                    if changeState.conPass < changeState.conTotal
                        if isempty(con_V)
                            changeState.conPass = changeState.conTotal;
                        else
                            idx = changeState.conIdx + 1;
                            if idx > numel(con_V)
                                idx = 1;
                                changeState.conPass = changeState.conPass + 1;
                                if changeState.conPass >= changeState.conTotal
                                    changeState.conIdx = 0;
                                    continue;
                                end
                            end
                            changeState.conIdx = idx;
                            drawnow();
                            Population = EAGO.ConvergenceOptimization_R(Problem,Population,con_V(idx),R,c_fmin,c_fmax);
                            continue;
                        end
                    end

                    if changeState.distPass < changeState.distTotal
                        if isempty(Div_V)
                            changeState.distPass = changeState.distTotal;
                        else
                            drawnow();
                            Population = EAGO.DistributionOptimization(Problem,Population,Div_V,R,c_fmin,c_fmax);
                            changeState.distPass = changeState.distPass + 1;
                            continue;
                        end
                    end

                    change = false;
                    changeState.initialized = false;
                    intensiveState.initialized = false;
                    continue;
                else
                    if ~intensiveState.initialized
                        if rank(Population.objs) == Problem.M
                            objs = Population.objs;
                            x1   = Population.decs;
                            R2   = objs\x1;
                            [Div_V2,con_V1,con_V_plus] = EAGO.VariableAnalysis2(Problem,Population,nSel,nPer,R1);
                            if length(Div_V2) <= length(Div_V1)
                                Div_V1 = Div_V2;
                                R1     = R2;
                            end
                        else
                            [Div_V2,con_V1,con_V_plus] = EAGO.VariableAnalysis2(Problem,Population,nSel,nPer,R1);
                            if length(Div_V2) <= length(Div_V1)
                                Div_V1 = Div_V2;
                            end
                        end
                        intensiveState.step = 0;
                        intensiveState.initialized = true;
                    end

                    if intensiveState.step < 35
                        drawnow();
                        Population = EAGO.ConvergenceOptimization_1(Problem,Population,con_V1,R1,c_fmin,c_fmax);
                        intensiveState.step = intensiveState.step + 1;
                        continue;
                    elseif intensiveState.step < 70
                        drawnow();
                        Population = EAGO.ConvergenceOptimization_2(Problem,Population,con_V_plus,R1,c_fmin,c_fmax);
                        intensiveState.step = intensiveState.step + 1;
                        continue;
                    elseif intensiveState.step < 80
                        drawnow();
                        Population = EAGO.DistributionOptimization2(Problem,Population,Div_V1,R1,c_fmin,c_fmax);
                        intensiveState.step = intensiveState.step + 1;
                        continue;
                    else
                        change = true;
                        changeState.initialized = false;
                        intensiveState.initialized = false;
                        continue;
                    end
                end
            end
        end
    end

    methods(Static, Access = private)
        function [Div_V,con_V] = VariableClustering(Problem,Population,nSel,nPer)
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
            Div_V = find(~VariableKind);
            con_V = find(VariableKind);
        end

        function Population = ConvergenceOptimization_R(Problem,Population,con_V,R,c_fmin,c_fmax)
            [N,~] = size(Population.decs);
            if Problem.FE >= Problem.maxFE
                return;
            end

            OffDec = Population.decs;

            NewObjs  = EAGO.GAhalf3(Population(randperm(N)).objs,N,c_fmin,c_fmax);
            Offspring_Convergence  = NewObjs * R(:,con_V);

            NewObjs2 = EAGO.GAhalf2(Population(randperm(N)).objs,N);
            Offspring_Convergence2 = NewObjs2 * R(:,con_V);

            OffDec(:,con_V) = OffDec(randperm(N),con_V) + (Offspring_Convergence-Offspring_Convergence2)*0.5;
            OffDec(:,con_V) = min(max(OffDec(:,con_V),repmat(Problem.lower(con_V),N,1)),repmat(Problem.upper(con_V),N,1));

            remain = Problem.maxFE - Problem.FE;
            if remain <= 0
                return;
            elseif remain < N
                evalIdx = randperm(N,remain);
            else
                evalIdx = 1:N;
            end
            [Offspring,evalIdx] = EAGO.EvaluateWithBudget(Problem,OffDec,evalIdx);
            if isempty(evalIdx)
                return;
            end

            allCon = EAGO.calCon([Population.objs;Offspring.objs]);
            Con    = allCon(1:N);
            newCon = allCon(N+1:end);
            updated = Con(evalIdx) > newCon;
            Population(evalIdx(updated)) = Offspring(updated);
        end

        function Population2 = DistributionOptimization(Problem,Population,Div_V,R,c_fmin,c_fmax)
            N = length(Population);
            if Problem.FE >= Problem.maxFE
                Population2 = Population;
                return;
            end
            OffDec = Population(TournamentSelection(2,N,EAGO.calCon(Population.objs))).decs;

            NewObjs  = EAGO.GAhalf3(Population.objs,N,c_fmin,c_fmax);
            Offspring_Convergence  = NewObjs * R(:,Div_V);

            NewObjs2 = EAGO.GAhalf2(Population.objs,N);
            Offspring_Convergence2 = NewObjs2 * R(:,Div_V);

            temp  = sqrt(OffDec(randperm(N),Div_V).*OffDec(randperm(N),Div_V));
            temp2 = OffDec(:,Div_V);
            a     = randperm(N);
            temp(temp>OffDec(a,Div_V)) = temp2(temp>OffDec(a,Div_V));
            OffDec(:,Div_V) = temp + (Offspring_Convergence-Offspring_Convergence2)*0.5;
            OffDec(:,Div_V) = min(max(OffDec(:,Div_V),repmat(Problem.lower(Div_V),N,1)),repmat(Problem.upper(Div_V),N,1));

            [Offspring,~] = EAGO.EvaluateWithBudget(Problem,OffDec);
            if isempty(Offspring)
                Population2 = Population;
                return;
            end
            Population2 = EnvironmentalSelection([Population,Offspring],N);
        end

        function [Div_V,con_V,con_V_plus] = VariableAnalysis2(Problem,Population,nSel,nPer,R)
            VariableKind      = false(1,Problem.D);
            VariableKind_plus = false(1,Problem.D);
            for i = 1 : Problem.D
                if Problem.FE >= Problem.maxFE
                    break;
                end
                drawnow();
                if Problem.maxFE - Problem.FE < 3*nPer
                    break;
                end
                Sample = randi(Problem.N,1,nSel);
                result = zeros(1,nSel);
                for j = 1 : nSel
                    if Problem.FE >= Problem.maxFE || Problem.maxFE - Problem.FE < 3*nPer
                        break;
                    end

                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    Decs(:,i) = unifrnd(Problem.lower(i),Problem.upper(i),size(Decs,1),1);
                    newPopu_random = Problem.Evaluation(Decs).objs;
                    newPopu_random_average = sum(newPopu_random,1)/nPer;

                    NewObjs  = repmat(Population(Sample(j)).objs,nPer,1) - repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring = NewObjs * R(:,i);
                    Decs(:,i) = min(max(Offspring,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    newPopu_reflex = Problem.Evaluation(Decs).objs;
                    newPopu_reflex_average = sum(newPopu_reflex,1)/nPer;

                    NewObjs_plus  = repmat(Population(Sample(j)).objs,nPer,1) + repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring_plus = NewObjs_plus * R(:,i);
                    Decs(:,i) = min(max(Offspring_plus,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    if Problem.maxFE - Problem.FE < nPer
                        break;
                    end
                    newPopu_reflex_plus = Problem.Evaluation(Decs).objs;
                    newPopu_reflex_plus_average = sum(newPopu_reflex_plus,1)/nPer;

                    if sum(newPopu_reflex_average.*newPopu_reflex_average)*0.9 <= sum(newPopu_random_average.*newPopu_random_average) && ...
                            sum(newPopu_reflex_average.*newPopu_reflex_average) <= sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average)
                        result(j) = 1;
                    elseif sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average)*0.9 <= sum(newPopu_random_average.*newPopu_random_average) && ...
                            sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average) < sum(newPopu_reflex_average.*newPopu_reflex_average)
                        result(j) = 2;
                    end
                end

                if sum(result == 1) >= sum(result == 2) && sum(result == 1) >= sum(result == 0)
                    VariableKind(i) = true;
                elseif sum(result == 2) > sum(result == 1) && sum(result == 2) >= sum(result == 0)
                    VariableKind_plus(i) = true;
                end
            end

            Div_V     = find(~(VariableKind | VariableKind_plus));
            con_V     = find(VariableKind);
            con_V_plus = find(VariableKind_plus);
        end

        function [Div_V,con_V,con_V_plus] = VariableAnalysis1(Problem,Population,nSel,nPer,R)
            VariableKind      = false(1,Problem.D);
            VariableKind_plus = false(1,Problem.D);
            for i = 1 : Problem.D
                if Problem.FE >= Problem.maxFE
                    break;
                end
                drawnow();
                if Problem.maxFE - Problem.FE < 3*nPer
                    break;
                end
                Sample = randi(Problem.N,1,nSel);
                result = zeros(1,nSel);
                for j = 1 : nSel
                    if Problem.FE >= Problem.maxFE || Problem.maxFE - Problem.FE < 3*nPer
                        break;
                    end

                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    Decs(:,i) = unifrnd(Problem.lower(i),Problem.upper(i),size(Decs,1),1);
                    newPopu_random = Problem.Evaluation(Decs).objs;
                    newPopu_random_average = sum(newPopu_random,1)/nPer;

                    NewObjs  = repmat(Population(Sample(j)).objs,nPer,1) - repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring = NewObjs * R(:,i);
                    Decs(:,i) = min(max(Offspring,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    newPopu_reflex = Problem.Evaluation(Decs).objs;
                    newPopu_reflex_average = sum(newPopu_reflex,1)/nPer;

                    NewObjs_plus  = repmat(Population(Sample(j)).objs,nPer,1) + repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring_plus = NewObjs_plus * R(:,i);
                    Decs(:,i) = min(max(Offspring_plus,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    if Problem.maxFE - Problem.FE < nPer
                        break;
                    end
                    newPopu_reflex_plus = Problem.Evaluation(Decs).objs;
                    newPopu_reflex_plus_average = sum(newPopu_reflex_plus,1)/nPer;

                    if sum(newPopu_reflex_average.*newPopu_reflex_average)*0.9 <= sum(newPopu_random_average.*newPopu_random_average) && ...
                            sum(newPopu_reflex_average.*newPopu_reflex_average) <= sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average)
                        result(j) = 1;
                    elseif sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average)*0.9 <= sum(newPopu_random_average.*newPopu_random_average) && ...
                            sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average) < sum(newPopu_reflex_average.*newPopu_reflex_average)
                        result(j) = 2;
                    end
                end

                if sum(result == 1) >= sum(result == 2) && sum(result == 1) >= sum(result == 0)
                    VariableKind(i) = true;
                elseif sum(result == 2) > sum(result == 1) && sum(result == 2) >= sum(result == 0)
                    VariableKind_plus(i) = true;
                end
            end

            Div_V      = find(~(VariableKind | VariableKind_plus));
            con_V      = find(VariableKind);
            con_V_plus = find(VariableKind_plus);
        end

        function Population = ConvergenceOptimization_1(Problem,Population,con_V,R,c_fmin,c_fmax)
            N = Problem.N;
            if isempty(con_V)
                return;
            end
            OffDec = Population.decs;

            NewObjs  = EAGO.GAhalf3(Population(randperm(N)).objs,N,c_fmin,c_fmax);
            Offspring_Convergence  = NewObjs * R(:,con_V);
            OffDec(:,con_V) = OffDec(randperm(N),con_V) + Offspring_Convergence*0.5;

            [Offspring,evalIdx] = EAGO.EvaluateWithBudget(Problem,OffDec,1:N);
            if isempty(evalIdx)
                return;
            end
            allCon = EAGO.calCon([Population.objs;Offspring.objs]);
            Con    = allCon(1:N);
            newCon = allCon(N+1:end);
            updated = Con(evalIdx) > newCon;
            Population(evalIdx(updated)) = Offspring(updated);
        end

        function Population = ConvergenceOptimization_2(Problem,Population,con_V_plus,R,c_fmin,c_fmax)
            N = Problem.N;
            if isempty(con_V_plus)
                return;
            end
            OffDec = Population.decs;

            NewObjs  = EAGO.GAhalf3(Population(randperm(N)).objs,N,c_fmin,c_fmax);
            Offspring_Convergence  = NewObjs * R(:,con_V_plus);
            OffDec(:,con_V_plus) = OffDec(randperm(N),con_V_plus) + Offspring_Convergence*0.5;
            OffDec(:,con_V_plus) = min(max(OffDec(:,con_V_plus),repmat(Problem.lower(con_V_plus),N,1)),repmat(Problem.upper(con_V_plus),N,1));

            [Offspring,evalIdx] = EAGO.EvaluateWithBudget(Problem,OffDec,1:N);
            if isempty(evalIdx)
                return;
            end
            allCon = EAGO.calCon([Population.objs;Offspring.objs]);
            Con    = allCon(1:N);
            newCon = allCon(N+1:end);
            updated = Con(evalIdx) > newCon;
            Population(evalIdx(updated)) = Offspring(updated);
        end

        function Population = DistributionOptimization2(Problem,Population,Div_V,R,c_fmin,c_fmax)
            N = Problem.N;
            if isempty(Div_V)
                return;
            end
            OffDec = Population.decs;

            NewObjs  = EAGO.GAhalf3(Population(randperm(N)).objs,N,c_fmin,c_fmax);
            Offspring_Convergence  = NewObjs * R(:,Div_V);

            NewObjs2 = EAGO.GAhalf2(Population(randperm(N)).objs,N);
            Offspring_Convergence2 = NewObjs2 * R(:,Div_V);

            temp  = sqrt(OffDec(randperm(N),Div_V).*OffDec(randperm(N),Div_V));
            temp2 = OffDec(:,Div_V);
            a     = randperm(N);
            temp(temp>OffDec(a,Div_V)) = temp2(temp>OffDec(a,Div_V));
            OffDec(:,Div_V) = temp + (Offspring_Convergence-Offspring_Convergence2)*0.5;
            OffDec(:,Div_V) = min(max(OffDec(:,Div_V),repmat(Problem.lower(Div_V),N,1)),repmat(Problem.upper(Div_V),N,1));

            [Offspring,~] = EAGO.EvaluateWithBudget(Problem,OffDec);
            if isempty(Offspring)
                return;
            end
            Population = EnvironmentalSelection([Population,Offspring],N);
        end

        function Offspring = GAhalf1(Parent,~)
            Offspring = Parent + Parent.*(rand(size(Parent)))*0.25;
        end

        function Offspring = GAhalf2(Parent,~)
            Offspring = Parent - Parent.*(rand(size(Parent)))*0.25;
        end

        function Offspring = GAhalf3(Parent,~,c_fmin,c_fmax)
            if nargin < 4
                c_fmin = 0.01;
                c_fmax = 10;
            end
            [proC,disC,proM,disM] = deal(1,20,1,20);

            lower = min(Parent,[],1)*c_fmin;
            upper = max(Parent,[],1)*c_fmax;

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

        function Offspring = GAhalf4(Parent,~)
            Offspring = Parent - Parent.*(rand(size(Parent))-0.5);
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
