function [CV,DV,CO] = VariableClustering(Global,Population,nSel,nPer)
% Detect the kind of each decision variable
   % 决策变量分类
% CV:convergence-related variables;
% DV:diversity-related variables;
% CO:贡献目标
% Detect the kind of each decision variable
    %% 
    [N,D] = size(Population.decs);
    ND    = NDSort(Population.objs,1) == 1;
    fmin  = min(Population(ND).objs,[],1);
    fmax  = max(Population(ND).objs,[],1);
    if any(fmax==fmin)
        fmax = ones(size(fmax));
        fmin = zeros(size(fmin));
    end
    %% Calculate the proper values of each decision variable
    Angle  = zeros(D,nSel);
    RMSE   = zeros(D,nSel);
    co   = zeros(D,nSel);%contribution object of decision variables
    Sample = randi(N,1,nSel);
    for i = 1 : D
        drawnow();
        % Generate several random solutions by perturbing the i-th dimension
        Decs      = repmat(Population(Sample).decs,nPer,1);
        %创建随机的连续均匀分布的数组
        Decs(:,i) = unifrnd(Global.lower(i),Global.upper(i),size(Decs,1),1); 
        newPopu   = INDIVIDUAL(Decs);
%         newpop=newPopu.objs;
%         plot3(newpop(:,1),newpop(:,2),newpop(:,3),'r*');
        for j = 1 : nSel
            % Normalize the objective values of the current perturbed solutions
            %第j个个体  扰动变量i的点集
            Points = newPopu(j:nSel:end).objs; 
            % Normalize 减最小除以scale-mean
            Points = (Points-repmat(fmin,size(Points,1),1))./repmat(fmax-fmin,size(Points,1),1);
            Points = Points - repmat(mean(Points,1),nPer,1);
            % Calculate the direction vector of the determining line
            [~,~,V] = svd(Points);
            Vector  = V(:,1)'./norm(V(:,1)');
            % 候选贡献目标co(i,j)
            [~,co(i,j)]= max(abs(Vector));
            % Calculate the root mean square error
            error = zeros(1,nPer);
            for k = 1  : nPer
                error(k) = norm(Points(k,:)-sum(Points(k,:).*Vector)*Vector);
            end
            RMSE(i,j) = sqrt(sum(error.^2));
            % Calculate the angle between the line and the hyperplane
            normal     = ones(1,size(Vector,2));
            sine       = abs(sum(Vector.*normal,2))./norm(Vector)./norm(normal);
            Angle(i,j) = real(asin(sine)/pi*180);
        end
    end
    %% Detect the kind of each decision variable
    VariableKind = (mean(RMSE,2)<1e-2)';
    result       = kmeans(Angle,2,'emptyaction','singleton')';
    if any(result(VariableKind)==1) && any(result(VariableKind)==2)
        if mean(mean(Angle(result==1&VariableKind,:))) > mean(mean(Angle(result==2&VariableKind,:)))
            VariableKind = VariableKind & result==1;
        else
            VariableKind = VariableKind & result==2;
        end
    end
    DV = find(~VariableKind);
    CV = find(VariableKind);
    for i = 1 : length(CV)
        CO{CV(i)}=[];
        t = tabulate(co(CV(i),:));
        [tn,~] = size(t);
        for m = 1:tn
           if t(m,2) ~=0
               CO{CV(i)}=[CO{CV(i)},m];
           end
        end
    end
end