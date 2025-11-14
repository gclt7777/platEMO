function [CV,DV,CO] = VariableClustering(Global,Population,nSel,nPer)
% Detect the type of each decision variable

    [N,D] = size(Population.decs);
    ND    = NDSort(Population.objs,1) == 1;
    fmin  = min(Population(ND).objs,[],1);
    fmax  = max(Population(ND).objs,[],1);
    if any(fmax==fmin)
        fmax = ones(size(fmax));
        fmin = zeros(size(fmin));
    end
    Angle     = zeros(D,nSel);
    RMSE      = zeros(D,nSel);
    co        = zeros(D,nSel);
    Sample    = randi(N,1,nSel);
    processed = false(1,D);
    for i = 1 : D
        drawnow();
        remainFE = Global.evaluation - Global.evaluated;
        maxPer   = floor(remainFE/max(1,nSel));
        if maxPer <= 0
            break;
        end
        currPer = min(nPer,maxPer);
        Decs      = repmat(Population(Sample).decs,currPer,1);
        Decs(:,i) = unifrnd(Global.lower(i),Global.upper(i),size(Decs,1),1);
        newPopu   = INDIVIDUAL(Decs);
        for j = 1 : nSel
            Points = newPopu(j:nSel:end).objs;
            Points = (Points-repmat(fmin,size(Points,1),1))./repmat(fmax-fmin,size(Points,1),1);
            Points = Points - repmat(mean(Points,1),size(Points,1),1);
            [~,~,V] = svd(Points,'econ');
            Vector  = V(:,1)'./norm(V(:,1)');
            [~,co(i,j)] = max(abs(Vector));
            perturbNum = size(Points,1);
            error = zeros(1,perturbNum);
            for k = 1 : perturbNum
                error(k) = norm(Points(k,:)-sum(Points(k,:).*Vector)*Vector);
            end
            RMSE(i,j) = sqrt(sum(error.^2));
            normal     = ones(1,size(Vector,2));
            sine       = abs(sum(Vector.*normal,2))./norm(Vector)./norm(normal);
            Angle(i,j) = real(asin(sine)/pi*180);
        end
        processed(i) = true;
    end
    processedIdx = find(processed);
    VariableKind = false(1,D);
    if ~isempty(processedIdx)
        VariableKind(processedIdx) = mean(RMSE(processedIdx,:),2)<1e-2;
        result = zeros(1,D);
        if length(processedIdx) >= 2
            tempResult = kmeans(Angle(processedIdx,:),2,'emptyaction','singleton')';
        else
            tempResult = ones(1,length(processedIdx));
        end
        result(processedIdx) = tempResult;
        if any(result(VariableKind)==1) && any(result(VariableKind)==2)
            if mean(mean(Angle(result==1 & VariableKind,:))) > mean(mean(Angle(result==2 & VariableKind,:)))
                VariableKind = VariableKind & result==1;
            else
                VariableKind = VariableKind & result==2;
            end
        end
    end
    DV = find(~VariableKind);
    CV = find(VariableKind);
    CO = cell(1,D);
    for i = 1 : length(CV)
        CO{CV(i)} = [];
        t = tabulate(co(CV(i),:));
        for m = 1 : size(t,1)
            if t(m,2) ~= 0
                CO{CV(i)} = [CO{CV(i)},t(m,1)]; %#ok<AGROW>
            end
        end
    end
end