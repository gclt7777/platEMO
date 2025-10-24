classdef EAGO_ABL < ALGORITHM
% <2023> <multi/many> <real/integer/label/binary/permutation>
% <algorithm> <L>
% EAGO with ablation switches for Algorithm5 (VariableClustering), VA1, VA2.
%
% 参数通过 Algorithm.ParameterSet 传入：
% [nSel,nPer,c_fmin,c_fmax, useAlg5, useVA1, useVA2, splitMode, splitRatio, seed]
%   useAlg5   : 1=用 Algorithm5(VariableClustering)，0=关闭并用后备分组
%   useVA1    : 1=用 VariableAnalysis1，0=关闭（继承 Alg5 的两组）
%   useVA2    : 1=用 VariableAnalysis2，0=关闭（维持 VA1 分组）
%   splitMode : 1=随机分组，2=T-能量分组（建议在 Alg5 关闭时使用）
%   splitRatio: Alg5 关闭时，分到 con_V 的比例
%   seed      : 随机种子
%
% 示例：
% main('-algorithm',@EAGO_ABL,'-problem',@DTLZ2,'-M',3,'-D',10,'-N',92,'-evaluation',5e4, ...
%      '-parameter',{5,30,0.01,10, 1,0,1, 2,0.5,42});

    methods
        function main(Algorithm, Problem)
            %% 参数
            [nSel,nPer,c_fmin,c_fmax, useAlg5, useVA1, useVA2, splitMode, splitRatio, seed] = ...
                Algorithm.ParameterSet(5,30,0.01,10, 0,1,1, 2,0.5,42); %#ok<ASGLU>
            rng(seed,'twister');

            %% 初始化
            Population = Problem.Initialization();
            Y = Population.objs; X = Population.decs;
            R  = Y \ X;      % OLS: T = Y\X
            R1 = R;

            % ===== 变量粗分：Alg5 或后备分组 =====
            if useAlg5
                % 注意：你本地的 VariableClustering 若依赖 INDIVIDUAL，会报错；
                % 若报错，请改为 useAlg5=0（默认）或把 VariableClustering 改成用 Problem.Evaluation。
                [Div_V, con_V] = VariableClustering(Problem,Population,nSel,nPer);
            else
                [Div_V, con_V] = EAGO_ABL.FallbackVarSplit(Problem, Population, splitMode, splitRatio);
            end

            % ===== VA1 或继承粗分 =====
            if useVA1
                [Div_V1, con_V1, con_V_plus] = EAGO_ABL.VariableAnalysis1(Problem,Population,nSel,nPer,R1);
            else
                Div_V1     = Div_V;
                con_V1     = con_V;
                con_V_plus = [];
            end

            change = true;
            while Algorithm.NotTerminated(Population)
                % —— 用 FE/maxFE 粗分阶段；若问题类没有这俩字段，就回退
                try
                    progress = Problem.FE / Problem.maxFE;
                catch
                    progress = 0;
                end
                temp = (progress < 0.9) * 10 + (progress >= 0.9) * 1;

                if change
                    % ===== 阶段A：两类算子（con_V / Div_V）=====
                    for j = 1 : temp
                        for ij = 1 : length(con_V)
                            drawnow();
                            Population = EAGO_ABL.ConvergenceOptimization_R(Population,con_V(ij),R,Problem);
                        end
                    end
                    for j = 1 : temp
                        drawnow();
                        Population = EAGO_ABL.DistributionOptimization(Population,Div_V,R,Problem);
                    end
                    change = false;

                else
                    % ===== 阶段B：可选 VA2 + 三类算子 =====
                    if useVA2
                        if rank(Population.objs) == Problem.M
                            Y = Population.objs; X = Population.decs;
                            R2 = Y \ X;    % OLS 更新
                            [Div_V2, con_V1_new, con_V_plus_new] = EAGO_ABL.VariableAnalysis2(Problem,Population,nSel,nPer,R1);
                            if length(Div_V2) <= length(Div_V1)
                                Div_V1     = Div_V2;
                                con_V1     = con_V1_new;
                                con_V_plus = con_V_plus_new;
                                R1         = R2;
                            end
                        else
                            [Div_V2, con_V1_new, con_V_plus_new] = EAGO_ABL.VariableAnalysis2(Problem,Population,nSel,nPer,R1);
                            if length(Div_V2) <= length(Div_V1)
                                Div_V1     = Div_V2;
                                con_V1     = con_V1_new;
                                con_V_plus = con_V_plus_new;
                            end
                        end
                    end

                    for j = 1 : 1
                        for i = 1 : 35
                            drawnow();
                            if ~isempty(con_V1)
                                Population = EAGO_ABL.ConvergenceOptimization_1(Population,con_V1,R1,Problem);
                            end
                        end
                        for i = 1 : 35
                            drawnow();
                            if ~isempty(con_V_plus)
                                Population = EAGO_ABL.ConvergenceOptimization_2(Population,con_V_plus,R1,Problem);
                            end
                        end
                        for i = 1 : 10
                            drawnow();
                            if ~isempty(Div_V1)
                                Population = EAGO_ABL.DistributionOptimization2(Population,Div_V1,R1,Problem);
                            end
                        end
                    end
                    change = true;
                end
            end
        end
    end

    %% ===================== private helpers =====================
    methods(Static, Access=private)
        function [Div_V, con_V] = FallbackVarSplit(Problem, Population, splitMode, splitRatio)
            D = Problem.D;
            splitRatio = max(0,min(1,splitRatio));
            k = max(0, min(D, round(splitRatio*D)));
            switch splitMode
                case 1
                    idx = randperm(D);
                    con_V = sort(idx(1:k));
                    Div_V = sort(idx(k+1:end));
                case 2
                    Y = Population.objs; X = Population.decs;
                    T = Y \ X;
                    en = sqrt(sum(T.^2,1));
                    [~,ord] = sort(en,'descend');
                    con_V = sort(ord(1:k));
                    Div_V = sort(ord(k+1:end));
                otherwise
                    con_V = [];
                    Div_V = 1:D;
            end
        end

        function Population = ConvergenceOptimization_R(Population,con_V,R,Problem)
            N = size(Population.decs,1);
            OffDec = Population.decs;
            NewObjs  = EAGO_ABL.GAhalf3(Population(randperm(N)).objs,N);
            NewObjs2 = EAGO_ABL.GAhalf2(Population(randperm(N)).objs,N);
            OC  = NewObjs  * R(:,con_V);
            OC2 = NewObjs2 * R(:,con_V);
            OffDec(:,con_V) = OffDec(randperm(N),con_V) + 0.5*(OC - OC2);
            OffDec(:,con_V) = min(max(OffDec(:,con_V),repmat(Problem.lower(con_V),N,1)),repmat(Problem.upper(con_V),N,1));
            Offspring = Problem.Evaluation(OffDec);
            allCon  = EAGO_ABL.calCon([Population.objs;Offspring.objs]);
            Con     = allCon(1:N);
            newCon  = allCon(N+1:end);
            updated = Con > newCon;
            Population(updated) = Offspring(updated);
        end

        function Population = DistributionOptimization(Population,Div_V,R,Problem)
            N      = length(Population);
            OffDec = Population(TournamentSelection(2,N,EAGO_ABL.calCon(Population.objs))).decs;
            NewObjs  = EAGO_ABL.GAhalf3(Population.objs,N);
            NewObjs2 = EAGO_ABL.GAhalf2(Population.objs,N);
            OC  = NewObjs  * R(:,Div_V);
            OC2 = NewObjs2 * R(:,Div_V);
            temp  = sqrt(OffDec(randperm(N),Div_V).*OffDec(randperm(N),Div_V));
            temp2 = OffDec(:,Div_V);
            a = randperm(N);
            temp(temp > OffDec(a,Div_V)) = temp2(temp > OffDec(a,Div_V));
            OffDec(:,Div_V) = temp + 0.5*(OC - OC2);
            OffDec(:,Div_V) = min(max(OffDec(:,Div_V),repmat(Problem.lower(Div_V),N,1)),repmat(Problem.upper(Div_V),N,1));
            Offspring = Problem.Evaluation(OffDec);
            Population = EnvironmentalSelection([Population,Offspring],N);
        end

        % ====== VA2：用 Problem.Evaluation 评估 ======
        function [Div_V,con_V,con_V_plus] = VariableAnalysis2(Problem,Population,nSel,nPer,R)
            VariableKind      = false(1,Problem.D);
            VariableKind_plus = false(1,Problem.D);
            for i = 1 : Problem.D
                drawnow();
                Sample = randi(Problem.N,1,nSel);
                result = zeros(1,nSel);
                for j = 1 : nSel
                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    % 随机化第 i 维
                    Decs(:,i) = unifrnd(Problem.lower(i),Problem.upper(i),size(Decs,1),1);
                    newPopu_random = Problem.Evaluation(Decs);
                    newPopu_random_average = sum(newPopu_random.objs,1)/nPer;

                    % minus（收缩）
                    NewObjs  = repmat(Population(Sample(j)).objs,nPer,1) - repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring = NewObjs * R(:,i);
                    Decs(:,i) = min(max(Offspring,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    newPopu_reflex = Problem.Evaluation(Decs);
                    newPopu_reflex_average = sum(newPopu_reflex.objs,1)/nPer;

                    % plus（扩张）
                    NewObjs_plus  = repmat(Population(Sample(j)).objs,nPer,1) + repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring_plus = NewObjs_plus * R(:,i);
                    Decs(:,i) = min(max(Offspring_plus,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    newPopu_reflex_plus = Problem.Evaluation(Decs);
                    newPopu_reflex_plus_average = sum(newPopu_reflex_plus.objs,1)/nPer;

                    if (sum(newPopu_reflex_average.*newPopu_reflex_average)*0.9 <= sum(newPopu_random_average.*newPopu_random_average) && ...
                        sum(newPopu_reflex_average.*newPopu_reflex_average) <= sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average))
                        result(j) = 1;
                    elseif (sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average)*0.9 <= sum(newPopu_random_average.*newPopu_random_average) && ...
                            sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average) < sum(newPopu_reflex_average.*newPopu_reflex_average))
                        result(j) = 2;
                    end
                end
                if sum(result == 1) >= sum(result == 2) && sum(result == 1) >= sum(result == 0)
                    VariableKind(i) = true;
                elseif (sum(result == 2) > sum(result == 1) && sum(result == 2) >= sum(result == 0))
                    VariableKind_plus(i) = true;
                end
            end
            Div_V = find(~(VariableKind | VariableKind_plus));
            con_V = find(VariableKind);
            con_V_plus = find(VariableKind_plus);
        end

        % ====== VA1：同上，用 Problem.Evaluation ======
        function [Div_V,con_V,con_V_plus] = VariableAnalysis1(Problem,Population,nSel,nPer,R)
            VariableKind      = false(1,Problem.D);
            VariableKind_plus = false(1,Problem.D);
            for i = 1 : Problem.D
                drawnow();
                Sample = randi(Problem.N,1,nSel);
                result = zeros(1,nSel);
                for j = 1 : nSel
                    Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                    Decs(:,i) = unifrnd(Problem.lower(i),Problem.upper(i),size(Decs,1),1);
                    newPopu_random = Problem.Evaluation(Decs);
                    newPopu_random_average = sum(newPopu_random.objs,1)/nPer;

                    NewObjs  = repmat(Population(Sample(j)).objs,nPer,1) - repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring = NewObjs * R(:,i);
                    Decs(:,i) = min(max(Offspring,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    newPopu_reflex = Problem.Evaluation(Decs);
                    newPopu_reflex_average = sum(newPopu_reflex.objs,1)/nPer;

                    NewObjs_plus  = repmat(Population(Sample(j)).objs,nPer,1) + repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer,Problem.M))*0.25;
                    Offspring_plus = NewObjs_plus * R(:,i);
                    Decs(:,i) = min(max(Offspring_plus,repmat(Problem.lower(i),nPer,1)),repmat(Problem.upper(i),nPer,1));
                    newPopu_reflex_plus = Problem.Evaluation(Decs);
                    newPopu_reflex_plus_average = sum(newPopu_reflex_plus.objs,1)/nPer;

                    if (sum(newPopu_reflex_average.*newPopu_reflex_average) <= sum(newPopu_random_average.*newPopu_random_average) && ...
                        sum(newPopu_reflex_average.*newPopu_reflex_average) <= sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average))
                        result(j) = 1;
                    elseif (sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average) <= sum(newPopu_random_average.*newPopu_random_average) && ...
                            sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average) < sum(newPopu_reflex_average.*newPopu_reflex_average))
                        result(j) = 2;
                    end
                end
                if sum(result == 1) >= sum(result == 2) && sum(result == 1) >= sum(result == 0)
                    VariableKind(i) = true;
                elseif (sum(result == 2) > sum(result == 1) && sum(result == 2) >= sum(result == 0))
                    VariableKind_plus(i) = true;
                end
            end
            Div_V = find(~(VariableKind | VariableKind_plus));
            con_V = find(VariableKind);
            con_V_plus = find(VariableKind_plus);
        end

        function Population = ConvergenceOptimization_1(Population,con_V,R,Problem)
            N = size(Population.decs,1);
            OffDec = Population.decs;
            NewObjs = EAGO_ABL.GAhalf2(Population.objs,N);
            Xhat = NewObjs * R(:,con_V);
            NewDec = min(max(Xhat,repmat(Problem.lower(con_V),N,1)),repmat(Problem.upper(con_V),N,1));
            OffDec(:,con_V) = 0.5*(NewDec + OffDec(:,con_V));
            Offspring = Problem.Evaluation(OffDec);
            allCon  = EAGO_ABL.calCon([Population.objs;Offspring.objs]);
            Con     = allCon(1:N);
            newCon  = allCon(N+1:end);
            updated = Con > newCon;
            Population(updated) = Offspring(updated);
        end

        function Population = ConvergenceOptimization_2(Population,con_V_plus,R,Problem)
            N = size(Population.decs,1);
            OffDec = Population.decs;
            NewObjs = EAGO_ABL.GAhalf1(Population.objs,N);
            Xhat = NewObjs * R(:,con_V_plus);
            NewDec = min(max(Xhat,repmat(Problem.lower(con_V_plus),N,1)),repmat(Problem.upper(con_V_plus),N,1));
            OffDec(:,con_V_plus) = 0.5*(NewDec + OffDec(:,con_V_plus));
            Offspring = Problem.Evaluation(OffDec);
            allCon  = EAGO_ABL.calCon([Population.objs;Offspring.objs]);
            Con     = allCon(1:N);
            newCon  = allCon(N+1:end);
            updated = Con > newCon;
            Population(updated) = Offspring(updated);
        end

        function Population = DistributionOptimization2(Population,Div_V,R,Problem)
            N = length(Population);
            OffDec = Population(TournamentSelection(2,N,EAGO_ABL.calCon(Population.objs))).decs;
            NewObjs  = EAGO_ABL.GAhalf1(Population.objs,N);
            NewObjs2 = EAGO_ABL.GAhalf2(Population.objs,N);
            OC  = NewObjs  * R(:,Div_V);
            OC2 = NewObjs2 * R(:,Div_V);
            temp  = sqrt(OffDec(randperm(N),Div_V).*OffDec(randperm(N),Div_V));
            temp2 = OffDec(:,Div_V);
            a = randperm(N);
            temp(temp > OffDec(a,Div_V)) = temp2(temp > OffDec(a,Div_V));
            OffDec(:,Div_V) = temp + 0.5*(OC - OC2);
            OffDec(:,Div_V) = min(max(OffDec(:,Div_V),repmat(Problem.lower(Div_V),N,1)),repmat(Problem.upper(Div_V),N,1));
            Offspring = Problem.Evaluation(OffDec);
            Population = EnvironmentalSelection([Population,Offspring],N);
        end

        function Offspring = GAhalf1(Parent,~), Offspring = Parent + Parent.*(rand(size(Parent)))*0.25; end
        function Offspring = GAhalf2(Parent,~), Offspring = Parent - Parent.*(rand(size(Parent)))*0.25; end
        function Offspring = GAhalf3(Parent,~)
            c_fmin = 0.01; c_fmax = 10; [proC,disC,proM,disM] = deal(1,20,1,20);
            lower = min(Parent,[],1)*c_fmin; upper = max(Parent,[],1)*c_fmax;
            Parent1 = Parent(1:floor(end/2),:);
            Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
            [N,D]   = size(Parent1);
            beta = zeros(N,D); mu = rand(N,D);
            beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
            beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
            beta = beta.*(-1).^randi([0,1],N,D); beta(rand(N,D)<0.5) = 1;
            beta(repmat(rand(N,1)>proC,1,D)) = 1;
            Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                         (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
            Lower = repmat(lower,2*N,1); Upper = repmat(upper,2*N,1);
            Site  = rand(2*N,D) < proM/D; mu = rand(2*N,D);
            temp  = Site & mu<=0.5; Offspring = min(max(Offspring,Lower),Upper);
            Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).* ...
                (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
            temp = Site & mu>0.5;
            Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).* ...
                (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
        end

        function Con = calCon(PopuObj)
            FrontNo = NDSort(PopuObj,inf);
            Con     = sum(PopuObj,2);
            Con     = FrontNo'*(max(Con)-min(Con)) + Con;
        end
    end
end
