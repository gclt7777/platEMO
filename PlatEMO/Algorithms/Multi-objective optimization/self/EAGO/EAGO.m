classdef EAGO < ALGORITHM
    % <2023> <multi/many> <real/integer/label/binary/permutation>
    % Objective Space-based Population Generation (EAGO)
    % Deng et al., IEEE Transactions on Evolutionary Computation, 2023

    methods
        function main(Algorithm, Problem)
            %% Parameter setting
            [nSel,nPer,c_fmin,c_fmax] = Algorithm.ParameterSet(5,30,0.01,10);
            % [nSel,nPer,c_fmin,c_fmax] = Algorithm.ParameterSet(10,50,1,1);

            %% Initialization
            Population = Problem.Initialization();
            R  = Population.objs \ Population.decs;
            R1 = R;

            % Variable clustering and initial variable classification
            [Div_V,con_V] = VariableClustering(Problem,Population,nSel,nPer);
            [Div_V1,con_V1,con_V_plus] = EAGO.VariableAnalysis1(Problem,Population,nSel,nPer,R1);

            change = true;
            while Algorithm.NotTerminated(Population)
                % Follow original implementation: use FE ratio to adjust temp
                try
                    progress = Problem.FE / Problem.maxFE;
                catch
                    progress = 0;
                end
                temp = (progress < 0.9) * 10 + (progress >= 0.9) * 1;

                if change
                    % ----- convergence channel (Algorithm 5 groups) -----
                    for j = 1 : temp
                        for idx = 1 : length(con_V)
                            drawnow();
                            Population = EAGO.ConvergenceOptimization_R(Population,con_V(idx),R,Problem,c_fmin,c_fmax);
                        end
                    end
                    % ----- diversity channel -----
                    for j = 1 : temp
                        drawnow();
                        Population = EAGO.DistributionOptimization(Population,Div_V,R,Problem,c_fmin,c_fmax);
                    end
                    change = false;
                else
                    % ----- dynamic variable classification (Algorithm 4) -----
                    if rank(Population.objs,1e-10) == Problem.M
                        R2 = Population.objs \ Population.decs;
                        [Div_V2,con_V1_new,con_V_plus_new] = EAGO.VariableAnalysis2(Problem,Population,nSel,nPer,R1);
                        if length(Div_V2) <= length(Div_V1)
                            Div_V1     = Div_V2;
                            con_V1     = con_V1_new;
                            con_V_plus = con_V_plus_new;
                            R1         = R2;
                        end
                    else
                        [Div_V2,con_V1_new,con_V_plus_new] = EAGO.VariableAnalysis2(Problem,Population,nSel,nPer,R1);
                        if length(Div_V2) <= length(Div_V1)
                            Div_V1     = Div_V2;
                            con_V1     = con_V1_new;
                            con_V_plus = con_V_plus_new;
                        end
                    end

                    % ----- three-way optimization: decrease / increase / general -----
                    for i = 1 : 35
                        drawnow();
                        Population = EAGO.ConvergenceOptimization_1(Population,con_V1,R1,Problem);
                    end
                    for i = 1 : 35
                        drawnow();
                        Population = EAGO.ConvergenceOptimization_2(Population,con_V_plus,R1,Problem);
                    end
                    for i = 1 : 10
                        drawnow();
                        Population = EAGO.DistributionOptimization2(Population,Div_V1,R1,Problem,c_fmin,c_fmax);
                    end

                    change = true;
                end
            end
        end
    end

    methods(Static, Access = private)
        %% ===== Convergence / diversity operations =====
        function Population = ConvergenceOptimization_R(Population,con_V,R,Problem,c_fmin,c_fmax)
            if isempty(con_V), return; end
            N = size(Population.decs,1);
            OffDec = Population.decs;

            Parents = Population(randperm(N)).objs;
            NewObjs  = EAGO.GAhalf3(Parents,N,c_fmin,c_fmax);
            NewObjs2 = EAGO.GAhalf2(Population(randperm(N)).objs,N);

            Offspring_Convergence  = NewObjs  * R(:,con_V);
            Offspring_Convergence2 = NewObjs2 * R(:,con_V);

            OffDec(:,con_V) = OffDec(randperm(N),con_V) + 0.5*(Offspring_Convergence - Offspring_Convergence2);
            OffDec(:,con_V) = min(max(OffDec(:,con_V),repmat(Problem.lower(con_V),N,1)), ...
                                               repmat(Problem.upper(con_V),N,1));

            Offspring  = Problem.Evaluation(OffDec);
            allCon     = EAGO.calCon([Population.objs;Offspring.objs]);
            Con        = allCon(1:N);
            newCon     = allCon(N+1:end);
            updated    = Con > newCon;
            Population(updated) = Offspring(updated);
        end

        function Population = DistributionOptimization(Population,Div_V,R,Problem,c_fmin,c_fmax)
            if isempty(Div_V), return; end
            N      = length(Population);
            OffDec = Population(TournamentSelection(2,N,EAGO.calCon(Population.objs))).decs;

            NewObjs  = EAGO.GAhalf3(Population.objs,N,c_fmin,c_fmax);
            NewObjs2 = EAGO.GAhalf2(Population.objs,N);

            Offspring_Convergence  = NewObjs  * R(:,Div_V);
            Offspring_Convergence2 = NewObjs2 * R(:,Div_V);

            temp  = sqrt(OffDec(randperm(N),Div_V).*OffDec(randperm(N),Div_V));
            temp2 = OffDec(:,Div_V);
            a = randperm(N);
            temp(temp > OffDec(a,Div_V)) = temp2(temp > OffDec(a,Div_V));
            OffDec(:,Div_V) = temp + 0.5*(Offspring_Convergence - Offspring_Convergence2);
            OffDec(:,Div_V) = min(max(OffDec(:,Div_V),repmat(Problem.lower(Div_V),N,1)), ...
                                               repmat(Problem.upper(Div_V),N,1));

            Offspring  = Problem.Evaluation(OffDec);
            Population = EnvironmentalSelection([Population,Offspring], Problem.N);
        end

        function Population = ConvergenceOptimization_1(Population,con_V,R,Problem)
            if isempty(con_V), return; end
            N = size(Population.decs,1);
            OffDec = Population.decs;

            NewObjs = EAGO.GAhalf2(Population.objs,N);
            Offspring_Convergence = NewObjs * R(:,con_V);
            NewDec = min(max(Offspring_Convergence,repmat(Problem.lower(con_V),N,1)), ...
                                        repmat(Problem.upper(con_V),N,1));

            OffDec(:,con_V) = 0.5*(NewDec + OffDec(:,con_V));
            Offspring = Problem.Evaluation(OffDec);

            allCon  = EAGO.calCon([Population.objs;Offspring.objs]);
            Con     = allCon(1:N);
            newCon  = allCon(N+1:end);
            updated = Con > newCon;
            Population(updated) = Offspring(updated);
        end

        function Population = ConvergenceOptimization_2(Population,con_V_plus,R,Problem)
            if isempty(con_V_plus), return; end
            N = size(Population.decs,1);
            OffDec = Population.decs;

            NewObjs = EAGO.GAhalf1(Population.objs,N);
            Offspring_Convergence = NewObjs * R(:,con_V_plus);
            NewDec = min(max(Offspring_Convergence,repmat(Problem.lower(con_V_plus),N,1)), ...
                                        repmat(Problem.upper(con_V_plus),N,1));

            OffDec(:,con_V_plus) = 0.5*(NewDec + OffDec(:,con_V_plus));
            Offspring = Problem.Evaluation(OffDec);

            allCon  = EAGO.calCon([Population.objs;Offspring.objs]);
            Con     = allCon(1:N);
            newCon  = allCon(N+1:end);
            updated = Con > newCon;
            Population(updated) = Offspring(updated);
        end

        function Population = DistributionOptimization2(Population,Div_V,R,Problem,c_fmin,c_fmax)
            if isempty(Div_V), return; end
            N      = length(Population);
            OffDec = Population(TournamentSelection(2,N,EAGO.calCon(Population.objs))).decs;

            NewObjs  = EAGO.GAhalf1(Population.objs,N);
            NewObjs2 = EAGO.GAhalf2(Population.objs,N);

            Offspring_Convergence  = NewObjs  * R(:,Div_V);
            Offspring_Convergence2 = NewObjs2 * R(:,Div_V);

            temp  = sqrt(OffDec(randperm(N),Div_V).*OffDec(randperm(N),Div_V));
            temp2 = OffDec(:,Div_V);
            a = randperm(N);
            temp(temp > OffDec(a,Div_V)) = temp2(temp > OffDec(a,Div_V));
            OffDec(:,Div_V) = temp + 0.5*(Offspring_Convergence - Offspring_Convergence2);
            OffDec(:,Div_V) = min(max(OffDec(:,Div_V),repmat(Problem.lower(Div_V),N,1)), ...
                                               repmat(Problem.upper(Div_V),N,1));

            Offspring  = Problem.Evaluation(OffDec);
            Population = EnvironmentalSelection([Population,Offspring], Problem.N);
        end

        %% ===== Variable classification (Algorithm 4 style) =====
        function [Div_V,con_V,con_V_plus] = VariableAnalysis2(Problem,Population,nSel,nPer,R)
            VariableKind      = false(1,Problem.D);
            VariableKind_plus = false(1,Problem.D);
            for i = 1 : Problem.D
                drawnow();
                Sample = randi(Problem.N,1,nSel);
                result = zeros(1,nSel);
                for j = 1 : nSel
                    % Random perturbation (G)
                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    Decs(:,i) = unifrnd(Problem.lower(i),Problem.upper(i),nPer,1);
                    avg_random = mean(Problem.Evaluation(Decs).objs,1);

                    % Decrease (D)
                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    NewObjs   = repmat(Population(Sample(j)).objs,nPer,1) - repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring = NewObjs * R(:,i);
                    Decs(:,i) = min(max(Offspring,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    avg_ref   = mean(Problem.Evaluation(Decs).objs,1);

                    % Increase (I)
                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    NewObjs_p = repmat(Population(Sample(j)).objs,nPer,1) + repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring_p = NewObjs_p * R(:,i);
                    Decs(:,i) = min(max(Offspring_p,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    avg_ref_p = mean(Problem.Evaluation(Decs).objs,1);

                    if (sum(avg_ref.^2)*0.9 <= sum(avg_random.^2) && sum(avg_ref.^2) <= sum(avg_ref_p.^2))
                        result(j) = 1;    % prefer decrease
                    elseif (sum(avg_ref_p.^2)*0.9 <= sum(avg_random.^2) && sum(avg_ref_p.^2) < sum(avg_ref.^2))
                        result(j) = 2;    % prefer increase
                    end
                end
                if sum(result==1) >= sum(result==2) && sum(result==1) >= sum(result==0)
                    VariableKind(i) = true;
                elseif sum(result==2) > sum(result==1) && sum(result==2) >= sum(result==0)
                    VariableKind_plus(i) = true;
                end
            end
            Div_V = find(~(VariableKind | VariableKind_plus));
            con_V = find(VariableKind);
            con_V_plus = find(VariableKind_plus);
        end

        function [Div_V,con_V,con_V_plus] = VariableAnalysis1(Problem,Population,nSel,nPer,R)
            VariableKind      = false(1,Problem.D);
            VariableKind_plus = false(1,Problem.D);
            for i = 1 : Problem.D
                drawnow();
                Sample = randi(Problem.N,1,nSel);
                result = zeros(1,nSel);
                for j = 1 : nSel
                    % Random (G)
                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    Decs(:,i) = unifrnd(Problem.lower(i),Problem.upper(i),nPer,1);
                    avg_random = mean(Problem.Evaluation(Decs).objs,1);

                    % Decrease (D)
                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    NewObjs   = repmat(Population(Sample(j)).objs,nPer,1) - repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring = NewObjs * R(:,i);
                    Decs(:,i) = min(max(Offspring,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    avg_ref   = mean(Problem.Evaluation(Decs).objs,1);

                    % Increase (I)
                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    NewObjs_p = repmat(Population(Sample(j)).objs,nPer,1) + repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring_p = NewObjs_p * R(:,i);
                    Decs(:,i) = min(max(Offspring_p,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    avg_ref_p = mean(Problem.Evaluation(Decs).objs,1);

                    if (sum(avg_ref.^2) <= sum(avg_random.^2) && sum(avg_ref.^2) <= sum(avg_ref_p.^2))
                        result(j) = 1;
                    elseif (sum(avg_ref_p.^2) <= sum(avg_random.^2) && sum(avg_ref_p.^2) < sum(avg_ref.^2))
                        result(j) = 2;
                    end
                end
                if sum(result==1) >= sum(result==2) && sum(result==1) >= sum(result==0)
                    VariableKind(i) = true;
                elseif sum(result==2) > sum(result==1) && sum(result==2) >= sum(result==0)
                    VariableKind_plus(i) = true;
                end
            end
            Div_V = find(~(VariableKind | VariableKind_plus));
            con_V = find(VariableKind);
            con_V_plus = find(VariableKind_plus);
        end

        %% ===== Objective-space "operators" =====
        function Offspring = GAhalf1(Parent,N)
            Offspring = Parent + Parent.*(rand(size(Parent)))*0.25;
            if size(Offspring,1) > N, Offspring = Offspring(1:N,:); end
        end

        function Offspring = GAhalf2(Parent,N)
            Offspring = Parent - Parent.*(rand(size(Parent)))*0.25;
            if size(Offspring,1) > N, Offspring = Offspring(1:N,:); end
        end

        function Offspring = GAhalf3(Parent,N,c_fmin,c_fmax)
            [proC,disC,proM,disM] = deal(1,20,1,20);

            lower = min(Parent,[],1)*c_fmin;
            upper = max(Parent,[],1)*c_fmax;

            P1 = Parent(1:floor(end/2),:);
            P2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
            [Nh,D] = size(P1);

            beta = zeros(Nh,D);
            mu   = rand(Nh,D);
            beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
            beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
            beta = beta.*(-1).^randi([0,1],Nh,D);
            beta(rand(Nh,D)<0.5) = 1;
            beta(repmat(rand(Nh,1)>proC,1,D)) = 1;

            Offspring = [(P1+P2)/2 + beta.*(P1-P2)/2;
                         (P1+P2)/2 - beta.*(P1-P2)/2];

            Lower = repmat(lower,size(Offspring,1),1);
            Upper = repmat(upper,size(Offspring,1),1);
            Site  = rand(size(Offspring)) < proM/D;
            mu    = rand(size(Offspring));
            tmp   = Site & mu<=0.5;
            Offspring = min(max(Offspring,Lower),Upper);
            Offspring(tmp) = Offspring(tmp)+(Upper(tmp)-Lower(tmp)).* ...
                ((2.*mu(tmp)+(1-2.*mu(tmp)).*(1-(Offspring(tmp)-Lower(tmp))./(Upper(tmp)-Lower(tmp))).^(disM+1)).^(1/(disM+1))-1);
            tmp   = Site & mu>0.5;
            Offspring(tmp) = Offspring(tmp)+(Upper(tmp)-Lower(tmp)).* ...
                (1-(2.*(1-mu(tmp))+2.*(mu(tmp)-0.5).*(1-(Upper(tmp)-Offspring(tmp))./(Upper(tmp)-Lower(tmp))).^(disM+1)).^(1/(disM+1)));

            if size(Offspring,1) >= N
                Offspring = Offspring(1:N,:);
            else
                Offspring = Offspring(randi(size(Offspring,1),N,1),:);
            end
        end

        %% ===== Auxiliary: convergence metric =====
        function Con = calCon(PopuObj)
            FrontNo = NDSort(PopuObj,inf);
            Con     = sum(PopuObj,2);
            Con     = FrontNo'*(max(Con)-min(Con)) + Con;
        end
    end
end
