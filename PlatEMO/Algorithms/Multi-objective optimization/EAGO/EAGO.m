classdef EAGO < ALGORITHM
    % <2023> <multi/many> <real/integer/label/binary/permutation>
    % EAGO: Objective Space-based Population Generation
    % Deng et al., IEEE TEC, 2023

    methods
        function main(obj, Problem)
            %% 参数
            [nSel,nPer,c_fmin,c_fmax] = obj.ParameterSet(5,30,0.01,10);
            % [nSel,nPer,c_fmin,c_fmax] = obj.ParameterSet(10,50,1,1);

            %% 初始化
            Population = Problem.Initialization();
            objs = Population.objs;
            x1   = Population.decs;
            R  = objs \ x1;        % 等价 T = pinv(Y)*X
            R1 = R;

            % 变量聚类 + 初次变量定性
            [Div_V,con_V] = VariableClustering(Problem,Population,nSel,nPer);
            [Div_V1,con_V1,con_V_plus] = EAGO.VariableAnalysis1(Problem,Population,nSel,nPer,R1);

            change = true; gen = 1;

            %% 主循环
            while obj.NotTerminated(Population)

                % 偶数代或首次更新 R（更稳）
                if isempty(R) || mod(gen,2)==0
                    R = Population.objs \ Population.decs;
                end

                % 为了稳妥，这里固定 temp=1（如需更激进可调大）
                temp = 1;

                if change
                    % —— 收敛通道（按初次分组）
                    for j = 1:temp
                        for ij = 1:length(con_V)
                            Population = EAGO.ConvergenceOptimization_R(Population,con_V(ij),R,Problem,c_fmin,c_fmax);
                        end
                    end
                    % —— 多样性通道
                    for j = 1:temp
                        Population = EAGO.DistributionOptimization(Population,Div_V,R,Problem,c_fmin,c_fmax);
                    end
                    change = false;

                else
                    % —— 动态变量定性（使用 R1）
                    if rank(Population.objs,1e-10) == Problem.M
                        R2 = Population.objs \ Population.decs;
                        [Div_V2,con_V1,con_V_plus] = EAGO.VariableAnalysis2(Problem,Population,nSel,nPer,R1);
                        if length(Div_V2) <= length(Div_V1)
                            Div_V1 = Div_V2;  R1 = R2;
                        end
                    else
                        [Div_V2,con_V1,con_V_plus] = EAGO.VariableAnalysis2(Problem,Population,nSel,nPer,R1);
                        if length(Div_V2) <= length(Div_V1)
                            Div_V1 = Div_V2;
                        end
                    end

                    % —— 三路优化：D / I / G
                    for i = 1:35
                        Population = EAGO.ConvergenceOptimization_1(Population,con_V1,R1,Problem);      % decrease
                    end
                    for i = 1:35
                        Population = EAGO.ConvergenceOptimization_2(Population,con_V_plus,R1,Problem);  % increase
                    end
                    for i = 1:10
                        Population = EAGO.DistributionOptimization2(Population,Div_V1,R1,Problem,c_fmin,c_fmax);
                    end

                    change = true;
                end

                gen = gen + 1;
            end
        end
    end

    methods(Static, Access = private)
        %% ===== 收敛 / 多样性操作 =====
        function Population = ConvergenceOptimization_R(Population,con_V,R,Problem,c_fmin,c_fmax)
            % 目标空间：两组（更“强”的GAhalf3 vs “弱”的GAhalf2），取差分
            N = size(Population.decs,1);
            OffDec = Population.decs;

            NewObjs  = EAGO.GAhalf3(Population(randperm(N)).objs, N, c_fmin, c_fmax);
            NewObjs2 = EAGO.GAhalf2(Population(randperm(N)).objs, N);

            Offspring_Convergence  = NewObjs  * R(:,con_V);
            Offspring_Convergence2 = NewObjs2 * R(:,con_V);

            OffDec(:,con_V) = OffDec(randperm(N),con_V) + 0.5*(Offspring_Convergence - Offspring_Convergence2);
            OffDec(:,con_V) = min(max(OffDec(:,con_V),repmat(Problem.lower(con_V),N,1)), ...
                                               repmat(Problem.upper(con_V),N,1));

            Offspring  = Problem.Evaluation(OffDec);
            allCon     = EAGO.calCon([Population.objs; Offspring.objs]);
            Con        = allCon(1:N);
            newCon     = allCon(N+1:end);
            updated    = Con > newCon;
            Population(updated) = Offspring(updated);
        end

        function Population = DistributionOptimization(Population,Div_V,R,Problem,c_fmin,c_fmax)
            if isempty(Div_V), return; end
            N      = length(Population);
            OffDec = Population(TournamentSelection(2,N,EAGO.calCon(Population.objs))).decs;

            NewObjs  = EAGO.GAhalf3(Population.objs, N, c_fmin, c_fmax);
            NewObjs2 = EAGO.GAhalf2(Population.objs, N);

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
            % decrease 分支（D）
            if isempty(con_V), return; end
            N = size(Population.decs,1);
            OffDec = Population.decs;

            NewObjs = EAGO.GAhalf2(Population.objs, N);    % decrease
            Offspring_Convergence = NewObjs * R(:,con_V);
            NewDec = min(max(Offspring_Convergence,repmat(Problem.lower(con_V),N,1)), ...
                                        repmat(Problem.upper(con_V),N,1));

            OffDec(:,con_V) = 0.5*(NewDec + OffDec(:,con_V));   % 半步更稳
            Offspring = Problem.Evaluation(OffDec);

            allCon  = EAGO.calCon([Population.objs;Offspring.objs]);
            Con     = allCon(1:N);
            newCon  = allCon(N+1:end);
            updated = Con > newCon;
            Population(updated) = Offspring(updated);
        end

        function Population = ConvergenceOptimization_2(Population,con_V_plus,R,Problem)
            % increase 分支（I）
            if isempty(con_V_plus), return; end
            N = size(Population.decs,1);
            OffDec = Population.decs;

            NewObjs = EAGO.GAhalf1(Population.objs, N);    % increase
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

            NewObjs  = EAGO.GAhalf1(Population.objs, N);
            NewObjs2 = EAGO.GAhalf2(Population.objs, N);

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

        %% ===== 变量定性（Algorithm 4 风格） =====
        function [Div_V,con_V,con_V_plus] = VariableAnalysis2(Problem,Population,nSel,nPer,R)
            VariableKind      = false(1,Problem.D);
            VariableKind_plus = false(1,Problem.D);
            for i = 1:Problem.D
                Sample = randi(Problem.N,1,nSel);
                result = zeros(1,nSel);
                for j = 1:nSel
                    % —— 随机（G）
                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    Decs(:,i) = unifrnd(Problem.lower(i),Problem.upper(i),nPer,1);
                    avg_random = mean(Problem.Evaluation(Decs).objs,1);

                    % —— 减小（D）
                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    NewObjs   = repmat(Population(Sample(j)).objs,nPer,1) - repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring = NewObjs * R(:,i);
                    Decs(:,i) = min(max(Offspring,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    avg_ref   = mean(Problem.Evaluation(Decs).objs,1);

                    % —— 增大（I）
                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    NewObjs_p = repmat(Population(Sample(j)).objs,nPer,1) + repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring_p = NewObjs_p * R(:,i);
                    Decs(:,i) = min(max(Offspring_p,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    avg_ref_p = mean(Problem.Evaluation(Decs).objs,1);

                    if (sum(avg_ref.^2)*0.9 <= sum(avg_random.^2) && sum(avg_ref.^2) <= sum(avg_ref_p.^2))
                        result(j) = 1;    % 更倾向 decrease
                    elseif (sum(avg_ref_p.^2)*0.9 <= sum(avg_random.^2) && sum(avg_ref_p.^2) < sum(avg_ref.^2))
                        result(j) = 2;    % 更倾向 increase
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
            for i = 1:Problem.D
                Sample = randi(Problem.N,1,nSel);
                result = zeros(1,nSel);
                for j = 1:nSel
                    % —— 随机（G）
                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    Decs(:,i) = unifrnd(Problem.lower(i),Problem.upper(i),nPer,1);
                    avg_random = mean(Problem.Evaluation(Decs).objs,1);

                    % —— 减小（D）
                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    NewObjs   = repmat(Population(Sample(j)).objs,nPer,1) - repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring = NewObjs * R(:,i);
                    Decs(:,i) = min(max(Offspring,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    avg_ref   = mean(Problem.Evaluation(Decs).objs,1);

                    % —— 增大（I）
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

        %% ===== 目标空间“算子” =====
        function Offspring = GAhalf1(Parent,N)
            Offspring = Parent + Parent.*(rand(size(Parent)))*0.25;   % increase
            if size(Offspring,1) > N, Offspring = Offspring(1:N,:); end
        end

        function Offspring = GAhalf2(Parent,N)
            Offspring = Parent - Parent.*(rand(size(Parent)))*0.25;   % decrease
            if size(Offspring,1) > N, Offspring = Offspring(1:N,:); end
        end

        function Offspring = GAhalf3(Parent,N,c_fmin,c_fmax)
            % SBX + PM（在目标空间），返回 N×M
            [proC,disC,proM,disM] = deal(1,20,1,20);

            lower = min(Parent,[],1)*c_fmin;
            upper = max(Parent,[],1)*c_fmax;

            P1 = Parent(1:floor(end/2),:);
            P2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
            [Nh,D] = size(P1);

            % SBX
            beta = zeros(Nh,D);
            mu   = rand(Nh,D);
            beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
            beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
            beta = beta.*(-1).^randi([0,1],Nh,D);
            beta(rand(Nh,D)<0.5) = 1;
            beta(repmat(rand(Nh,1)>proC,1,D)) = 1;

            Offspring = [(P1+P2)/2 + beta.*(P1-P2)/2;
                         (P1+P2)/2 - beta.*(P1-P2)/2];

            % PM
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

            % 只保留 N 行
            if size(Offspring,1) >= N
                Offspring = Offspring(1:N,:);
            else
                Offspring = Offspring(randi(size(Offspring,1),N,1),:);
            end
        end

        %% ===== 辅助：排序度量 =====
        function Con = calCon(PopuObj)
            FrontNo = NDSort(PopuObj,inf);
            Con     = sum(PopuObj,2);
            Con     = FrontNo'*(max(Con)-min(Con)) + Con;
        end
    end
end
