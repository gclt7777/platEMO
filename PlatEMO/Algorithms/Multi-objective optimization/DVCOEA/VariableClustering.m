function [CV,DV,CO] = VariableClustering(Problem,Population,nSel,nPer,maxBudget)
% Detect the type of each decision variable
%
% CV: convergence-related variables
% DV: diversity-related variables
% CO: objectives contributed by each CV

    if nargin < 5
        maxBudget = inf;
    end
    % Limit the sampling effort so that preprocessing cannot exhaust the
    % evaluation budget.
    remaining      = max(0,min(maxBudget,Problem.maxFE - Problem.FE));
    if remaining <= 0
        CV = [];
        DV = 1:Problem.D;
        CO = cell(1,Problem.D);
        return;
    end

    % Cap the number of processed variables if the budget is too small to
    % touch every dimension.
    maxProcessVars = min(Problem.D,floor(remaining/max(1,nSel*nPer)));
    if maxProcessVars == 0
        CV = [];
        DV = 1:Problem.D;
        CO = cell(1,Problem.D);
        return;
    end
    chosenVars   = randperm(Problem.D,maxProcessVars);
    budgetPerVar = max(1,floor(remaining/maxProcessVars));
    nSel         = min(nSel,max(1,floor(sqrt(budgetPerVar))));
    nPer         = min(nPer,max(1,floor(budgetPerVar/nSel)));

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
    co     = zeros(D,nSel);
    Sample = randi(N,1,nSel);
    processed = false(1,D);
    startFE   = Problem.FE;
    for i = chosenVars
        if Problem.FE >= Problem.maxFE || Problem.FE - startFE >= maxBudget
            break;
        end
        processed(i) = true;
        drawnow();
        Decs      = repmat(Population(Sample).decs,nPer,1);
        Decs(:,i) = unifrnd(Problem.lower(i),Problem.upper(i),size(Decs,1),1);
        newPopu   = Problem.Evaluation(Decs);
        for j = 1 : nSel
            Points = newPopu(j:nSel:end).objs;
            Points = (Points-repmat(fmin,size(Points,1),1))./repmat(fmax-fmin,size(Points,1),1);
            Points = Points - repmat(mean(Points,1),nPer,1);
            [~,~,V] = svd(Points,'econ');
            Vector  = V(:,1)'./norm(V(:,1)');
            [~,co(i,j)] = max(abs(Vector));
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
    VariableKind = false(1,D);
    VariableKind(processed) = (mean(RMSE(processed,:),2)<1e-2)';
    result       = zeros(1,D);
    if any(processed)
        result(processed) = kmeans(Angle(processed,:),2,'emptyaction','singleton')';
    end
    if any(result(VariableKind)==1) && any(result(VariableKind)==2)
        if mean(mean(Angle(result==1&VariableKind,:))) > mean(mean(Angle(result==2&VariableKind,:)))
            VariableKind = VariableKind & result==1;
        else
            VariableKind = VariableKind & result==2;
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
