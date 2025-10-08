classdef EAGO < ALGORITHM
% <2025> <multi/many> <real/ineger>
% EAGO (PlatEMO class version; merged VariableClustering & EnvironmentalSelection)
% - 入口：main(Algorithm,Problem)
% - 评估：统一使用 Problem.Evaluation(...)（GUI FE 计数正确）
% - R = Y \ X 作为目标->决策的线性回映
% - 变量判断：VariableClustering + VariableAnalysis1/2
% - 三类优化：ConvergenceOptimization_R / _1 / _2，DistributionOptimization / _2
% - 选择：LMEA 风格角度截断（Truncation by angle）
% - 兼容修复：calCon（向量形状与逻辑标量）、自带 Tournament（避免 randi 报错）

methods
    function main(Algorithm,Problem)
        %% 参数（与原版一致）
        [nSel,nPer,c_fmin,c_fmax] = Algorithm.ParameterSet(5,30,0.01,10);

        %% 初始化
        Population = Problem.Initialization();
        objs = Population.objs;        % N×M
        x1   = Population.decs;        % N×D
        R    = objs\x1;                % M×D
        R1   = R;

        % 初始变量聚类（合并版 VariableClustering）
        [Div_V,con_V] = EAGO.VarCluster(Problem,Population,nSel,nPer);

        % 初始变量判别1
        [Div_V1,con_V1,con_V_plus] = EAGO.VariableAnalysis1(Problem,Population,nSel,nPer, R1);

        change = true;

        %% 进化循环
        while Algorithm.NotTerminated(Population)
            % temp 策略
            if Problem.FE / max(Problem.maxFE,1) < 0.9
                temp = 10;
            else
                temp = 1;
            end

            if change
                % —— 收敛优化（R）——
                for j = 1:temp
                    for ij = 1:length(con_V)
                        drawnow();
                        Population = EAGO.ConvergenceOptimization_R(Population,con_V(ij), R, Problem, c_fmin, c_fmax);
                    end
                end
                % —— 分布优化（Div_V）——
                for j = 1:temp
                    drawnow();
                    Population = EAGO.DistributionOptimization(Population,Div_V, R, Problem, c_fmin, c_fmax);
                end
                change = false;

            else
                % —— 动态变量判别2 + 更新 R1 —— 
                if rank(Population.objs) == Problem.M
                    objs = Population.objs; x1 = Population.decs;
                    R2   = objs\x1;
                    [Div_V2,con_V1,con_V_plus] = EAGO.VariableAnalysis2(Problem,Population,nSel,nPer, R1);
                    if length(Div_V2) <= length(Div_V1)
                        Div_V1 = Div_V2; R1 = R2;
                    end
                else
                    [Div_V2,con_V1,con_V_plus] = EAGO.VariableAnalysis2(Problem,Population,nSel,nPer, R1);
                    if length(Div_V2) <= length(Div_V1)
                        Div_V1 = Div_V2;
                    end
                end

                % —— 三类优化轮替 —— 
                for i = 1:35
                    drawnow();
                    Population = EAGO.ConvergenceOptimization_1(Population,con_V1, R1, Problem);
                end
                for i = 1:35
                    drawnow();
                    Population = EAGO.ConvergenceOptimization_2(Population,con_V_plus, R1, Problem);
                end
                for i = 1:10
                    drawnow();
                    Population = EAGO.DistributionOptimization2(Population,Div_V1, R1, Problem);
                end
                change = true;
            end
        end
    end
end

methods (Static, Access=private)
    %% ========= VariableClustering（合并自你提供的文件）=========
    function [PV,DV] = VarCluster(Problem,Population,nSel,nPer)
        [N,D] = size(Population.decs);
        ND    = NDSort(Population.objs,1) == 1;
        PopND = Population(ND);
        if isempty(PopND)
            fmin = min(Population.objs,[],1);
            fmax = max(Population.objs,[],1);
        else
            fmin = min(cat(1,PopND.objs),[],1);
            fmax = max(cat(1,PopND.objs),[],1);
        end
        if any(fmax==fmin)
            fmax = ones(size(fmax));
            fmin = zeros(size(fmin));
        end

        Angle = zeros(D,nSel);
        RMSE  = zeros(D,nSel);
        Sample = randi(N,1,nSel);
        for i = 1:D
            drawnow();
            % 仅第 i 维扰动
            Decs      = repmat(Population(Sample).decs,nPer,1);
            Decs(:,i) = Problem.lower(i) + (Problem.upper(i)-Problem.lower(i)).*rand(size(Decs,1),1);
            newPopu   = Problem.Evaluation(Decs);

            for j = 1:nSel
                Points = cat(1,newPopu(j:nSel:end).objs); % nPer×M
                Points = (Points - repmat(fmin,size(Points,1),1))./repmat(fmax-fmin,size(Points,1),1);
                Points = Points - repmat(mean(Points,1),nPer,1);
                [~,~,V] = svd(Points,'econ');
                Vector  = (V(:,1).'/norm(V(:,1)));
                err = zeros(1,nPer);
                for k = 1:nPer
                    err(k) = norm(Points(k,:)-sum(Points(k,:).*Vector)*Vector);
                end
                RMSE(i,j) = sqrt(sum(err.^2));
                normal     = ones(1,size(Vector,2));
                sine       = abs(sum(Vector.*normal,2))./norm(Vector)./norm(normal);
                Angle(i,j) = real(asin(sine)/pi*180);
            end
        end
        VariableKind = (mean(RMSE,2)<1e-2)';
        % kmeans 可能不存在：加 try/catch 退化为仅 RMSE 判别
        try
            result = kmeans(Angle,2)'; 
            if any(result(VariableKind)==1) && any(result(VariableKind)==2)
                if mean(mean(Angle(result==1&VariableKind,:))) > mean(mean(Angle(result==2&VariableKind,:)))
                    VariableKind = VariableKind & result==1;
                else
                    VariableKind = VariableKind & result==2;
                end
            end
        catch
            % 无统计工具箱时忽略角度聚类
        end
        PV = find(~VariableKind);   % 对应 Div_V
        DV = find(VariableKind);    % 对应 con_V
    end

    %% ========= VariableAnalysis1 / 2（改为用 Evaluation）=========
    function [Div_V,con_V, con_V_plus] = VariableAnalysis1(Problem,Population,nSel,nPer,R)
        VariableKind      = false(1,Problem.D);
        VariableKind_plus = false(1,Problem.D);
        for i = 1:Problem.D
            drawnow();
            Sample = randi(Problem.N,1,nSel);
            result = zeros(1,nSel);
            for j = 1:nSel
                % 随机
                Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                Decs(:,i) = Problem.lower(i) + (Problem.upper(i)-Problem.lower(i)).*rand(size(Decs,1),1);
                Rnd       = Problem.Evaluation(Decs);
                avgRnd    = mean(cat(1,Rnd.objs),1);

                % 反射（减）
                NewObjs   = repmat(Population(Sample(j)).objs,nPer,1) ...
                          - repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer, Problem.M))*0.25;
                Offs      = NewObjs * R(:, i);
                Decs(:,i) = EAGO.boundCols(Offs, Problem.lower(i), Problem.upper(i));
                Ref1      = Problem.Evaluation(Decs);
                avgRef1   = mean(cat(1,Ref1.objs),1);

                % 反射（加）
                NewObjsP  = repmat(Population(Sample(j)).objs,nPer,1) ...
                          + repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer, Problem.M))*0.25;
                OffsP     = NewObjsP * R(:, i);
                Decs(:,i) = EAGO.boundCols(OffsP, Problem.lower(i), Problem.upper(i));
                Ref2      = Problem.Evaluation(Decs);
                avgRef2   = mean(cat(1,Ref2.objs),1);

                if(sum(avgRef1.*avgRef1) <= sum(avgRnd.*avgRnd) && sum(avgRef1.*avgRef1) <= sum(avgRef2.*avgRef2))
                    result(j) = 1;
                elseif(sum(avgRef2.*avgRef2) <= sum(avgRnd.*avgRnd) && sum(avgRef2.*avgRef2) < sum(avgRef1.*avgRef1))
                    result(j) = 2;
                end
            end
            if sum(result==1) >= sum(result==2) && sum(result==1) >= sum(result==0)
                VariableKind(i) = true;
            elseif sum(result==2) > sum(result==1) && sum(result==2) >= sum(result==0)
                VariableKind_plus(i) = true;
            end
        end
        Div_V      = find(~(VariableKind | VariableKind_plus));
        con_V      = find(VariableKind);
        con_V_plus = find(VariableKind_plus);
    end

    function [Div_V,con_V, con_V_plus] = VariableAnalysis2(Problem,Population,nSel,nPer,R)
        VariableKind      = false(1,Problem.D);
        VariableKind_plus = false(1,Problem.D);
        for i = 1:Problem.D
            drawnow();
            Sample = randi(Problem.N,1,nSel);
            result = zeros(1,nSel);
            for j = 1:nSel
                % 随机
                Decs      = repmat(Population(Sample(j)).decs,nPer,1);
                Decs(:,i) = Problem.lower(i) + (Problem.upper(i)-Problem.lower(i)).*rand(size(Decs,1),1);
                Rnd       = Problem.Evaluation(Decs);
                avgRnd    = mean(cat(1,Rnd.objs),1);

                % 反射（减）
                NewObjs   = repmat(Population(Sample(j)).objs,nPer,1) ...
                          - repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer, Problem.M))*0.25;
                Offs      = NewObjs * R(:, i);
                Decs(:,i) = EAGO.boundCols(Offs, Problem.lower(i), Problem.upper(i));
                Ref1      = Problem.Evaluation(Decs);
                avgRef1   = mean(cat(1,Ref1.objs),1);

                % 反射（加）
                NewObjsP  = repmat(Population(Sample(j)).objs,nPer,1) ...
                          + repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer, Problem.M))*0.25;
                OffsP     = NewObjsP * R(:, i);
                Decs(:,i) = EAGO.boundCols(OffsP, Problem.lower(i), Problem.upper(i));
                Ref2      = Problem.Evaluation(Decs);
                avgRef2   = mean(cat(1,Ref2.objs),1);

                if(sum(avgRef1.*avgRef1)*0.9 <= sum(avgRnd.*avgRnd) && sum(avgRef1.*avgRef1) <= sum(avgRef2.*avgRef2))
                    result(j) = 1;
                elseif(sum(avgRef2.*avgRef2)*0.9 <= sum(avgRnd.*avgRnd) && sum(avgRef2.*avgRef2) < sum(avgRef1.*avgRef1))
                    result(j) = 2;
                end
            end
            if sum(result==1) >= sum(result==2) && sum(result==1) >= sum(result==0)
                VariableKind(i) = true;
            elseif sum(result==2) > sum(result==1) && sum(result==2) >= sum(result==0)
                VariableKind_plus(i) = true;
            end
        end
        Div_V      = find(~(VariableKind | VariableKind_plus));
        con_V      = find(VariableKind);
        con_V_plus = find(VariableKind_plus);
    end

    %% ========= 三类优化 =========
    function Population = ConvergenceOptimization_R(Population,con_idx, R, Problem, c_fmin, c_fmax)
        [N,~]  = size(Population.decs);
        OffDec = Population.decs;

        NewObjs  = EAGO.GAhalf3(Population(randperm(N)).objs, c_fmin, c_fmax);
        Off1     = NewObjs  * R(:, con_idx);
        NewObjs2 = EAGO.GAhalf2(Population(randperm(N)).objs, N);
        Off2     = NewObjs2 * R(:, con_idx);

        OffDec(:,con_idx) = OffDec(randperm(N),con_idx) + 0.5*(Off1 - Off2);
        OffDec(:,con_idx) = EAGO.boundCols(OffDec(:,con_idx), Problem.lower(con_idx), Problem.upper(con_idx));

        Offspring = Problem.Evaluation(OffDec);

        allObj = [Population.objs; Offspring.objs];
        Con    = EAGO.calCon(allObj);
        curC   = Con(1:N); newC = Con(N+1:end);
        upd    = curC > newC;
        Population(upd) = Offspring(upd);
    end

    function Population2 = DistributionOptimization(Population,Div_V, R, Problem, c_fmin, c_fmax)
        N      = length(Population);
        % 自带稳健 Tournament（替代 TournamentSelection）
        idxSel = EAGO.Tournament(2, N, EAGO.calCon(Population.objs));
        OffDec = Population(idxSel).decs;

        NewObjs  = EAGO.GAhalf3(Population.objs, c_fmin, c_fmax);
        Off1     = NewObjs  * R(:, Div_V);
        NewObjs2 = EAGO.GAhalf2(Population.objs, N);
        Off2     = NewObjs2 * R(:, Div_V);

        temp  = sqrt(OffDec(randperm(N),Div_V).*OffDec(randperm(N),Div_V));
        temp2 = OffDec(:,Div_V);
        a     = randperm(N);
        temp(temp>OffDec(a,Div_V)) = temp2(temp>OffDec(a,Div_V));
        OffDec(:,Div_V) = temp + 0.5*(Off1-Off2);
        OffDec(:,Div_V) = EAGO.boundCols(OffDec(:,Div_V), Problem.lower(Div_V), Problem.upper(Div_V));

        Offspring   = Problem.Evaluation(OffDec);
        Population2 = EAGO.EnvSelAngle([Population,Offspring], N);
    end

    function Population = ConvergenceOptimization_1(Population,con_V, R, Problem)
        [N,~]  = size(Population.decs);
        OffDec = Population.decs;

        NewObjs  = EAGO.GAhalf2(Population.objs, N);
        Off1     = NewObjs * R(:, con_V);
        NewDec   = EAGO.boundCols(Off1, Problem.lower(con_V), Problem.upper(con_V));

        OffDec(:,con_V) = (NewDec + OffDec(:,con_V))/2;
        Offspring = Problem.Evaluation(OffDec);

        allObj = [Population.objs; Offspring.objs];
        Con    = EAGO.calCon(allObj);
        curC   = Con(1:N); newC = Con(N+1:end);
        upd    = curC > newC;
        Population(upd) = Offspring(upd);
    end

    function Population = ConvergenceOptimization_2(Population,con_V_plus, R, Problem)
        [N,~]  = size(Population.decs);
        OffDec = Population.decs;

        NewObjs  = EAGO.GAhalf1(Population.objs, N);
        Off1     = NewObjs * R(:, con_V_plus);
        NewDec   = EAGO.boundCols(Off1, Problem.lower(con_V_plus), Problem.upper(con_V_plus));

        OffDec(:,con_V_plus) = (NewDec + OffDec(:,con_V_plus))/2;
        Offspring = Problem.Evaluation(OffDec);

        allObj = [Population.objs; Offspring.objs];
        Con    = EAGO.calCon(allObj);
        curC   = Con(1:N); newC = Con(N+1:end);
        upd    = curC > newC;
        Population(upd) = Offspring(upd);
    end

    function Population = DistributionOptimization2(Population,Div_V, R, Problem)
        N      = length(Population);
        idxSel = EAGO.Tournament(2, N, EAGO.calCon(Population.objs));
        OffDec = Population(idxSel).decs;

        NewObjs  = EAGO.GAhalf1(Population.objs, N);
        Off1     = NewObjs  * R(:, Div_V);
        NewObjs2 = EAGO.GAhalf2(Population.objs, N);
        Off2     = NewObjs2 * R(:, Div_V);

        temp  = sqrt(OffDec(randperm(N),Div_V).*OffDec(randperm(N),Div_V));
        temp2 = OffDec(:,Div_V);
        a     = randperm(N);
        temp(temp>OffDec(a,Div_V)) = temp2(temp>OffDec(a,Div_V));
        OffDec(:,Div_V) = temp + 0.5*(Off1-Off2);
        OffDec(:,Div_V) = EAGO.boundCols(OffDec(:,Div_V), Problem.lower(Div_V), Problem.upper(Div_V));

        Offspring  = Problem.Evaluation(OffDec);
        Population = EAGO.EnvSelAngle([Population,Offspring],N);
    end

    %% ========= 遗传算子 =========
    function Offspring = GAhalf1(Parent, ~)
        Offspring = Parent + Parent.*(rand(size(Parent)))*0.25;
    end
    function Offspring = GAhalf2(Parent, ~)
        Offspring = Parent - Parent.*(rand(size(Parent)))*0.25;
    end
    function Offspring = GAhalf3(Parent, c_fmin, c_fmax)
        [proC,disC,proM,disM] = deal(1,20,1,20);
        lower = min(Parent, [], 1)*c_fmin;
        upper = max(Parent, [], 1)*c_fmax;

        Parent1   = Parent(1:floor(end/2),:);
        Parent2   = Parent(floor(end/2)+1:floor(end/2)*2,:);
        [N,D]     = size(Parent1);

        % SBX
        beta = zeros(N,D);
        mu   = rand(N,D);
        beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
        beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
        beta = beta.*(-1).^randi([0,1],N,D);
        beta(rand(N,D)<0.5) = 1;
        beta(repmat(rand(N,1)>proC,1,D)) = 1;
        Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                     (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];

        % PM
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
    function Offspring = GAhalf4(Parent, ~)
        Offspring = Parent - Parent.*(rand(size(Parent))-0.5);
    end

    %% ========= 选择（LMEA EnvironmentalSelection + 角度截断）=========
    function Population = EnvSelAngle(Population,N)
        [FrontNo,MaxFNo] = NDSort(Population.objs,N);
        Next = FrontNo < MaxFNo;

        Last = find(FrontNo==MaxFNo);
        need = N - sum(Next);
        if need > 0 && ~isempty(Last)
            PopObjLast = cat(1,Population(Last).objs);
            Choose = EAGO.TruncationAngle(PopObjLast, need);
            Next(Last(Choose)) = true;
        end
        Population = Population(Next);
    end

    function Choose = TruncationAngle(PopObj,K)
        fmax   = max(PopObj,[],1);
        fmin   = min(PopObj,[],1);
        rng    = max(fmax-fmin, 1e-12);
        PopObj = (PopObj-repmat(fmin,size(PopObj,1),1))./repmat(rng,size(PopObj,1),1);
        Cosine = 1 - pdist2(PopObj,PopObj,'cosine');
        Cosine(logical(eye(length(Cosine)))) = 0;
        Choose = false(1,size(PopObj,1)); 
        [~,extreme] = max(PopObj,[],1);
        Choose(extreme) = true;
        if sum(Choose) > K
            selected = find(Choose);
            Choose   = selected(randperm(length(selected),K));
        else
            while sum(Choose) < K
                unSelected = find(~Choose);
                if isempty(unSelected), break; end
                mm = max(Cosine(~Choose,Choose),[],2);
                [~,x] = min(mm);
                Choose(unSelected(x)) = true;
            end
        end
    end

    %% ========= 工具 =========
    function A = boundCols(A, lb, ub)
        lb = lb(:)'; ub = ub(:)';
        if size(A,2)~=numel(lb)
            lb = repmat(lb(1),1,size(A,2));
            ub = repmat(ub(1),1,size(A,2));
        end
        A = min(max(A,repmat(lb,size(A,1),1)), repmat(ub,size(A,1),1));
    end

    function Con = calCon(PopuObj)
        % 收敛度：FrontNo 主导 + 目标和
        FrontNo = NDSort(PopuObj,inf);
        SumObj  = sum(PopuObj,2);
        f = FrontNo(:); s = SumObj(:);
        w = max(s) - min(s);
        if isempty(w), w = 1; else, w = w(1); end
        if any(~isfinite(w)) || w <= 0, w = 1; end
        Con = f.*w + s;  % N×1
    end

    function idx = Tournament(K, N, fit)
        % 稳健锦标赛选择（最小化）
        fit = fit(:);
        M = numel(fit);
        if M <= 0
            idx = ones(1, max(1,N));
            return;
        end
        bad = ~isfinite(fit);
        fit(bad) = inf;
        cand = randi(M, K, N);
        [~, bestRow] = min(fit(cand), [], 1);
        lin = sub2ind([K, N], bestRow, 1:N);
        idx = cand(lin);
    end
end
end
