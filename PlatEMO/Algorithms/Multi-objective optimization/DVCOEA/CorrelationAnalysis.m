function CVSet = CorrelationAnalysis(Problem,Population,CV,nCor,maxBudget)
% Detect the group of each distance variable
    if nargin < 5
        maxBudget = inf;
    end
    CVSet = {};
    % Limit the correlation tests based on the remaining evaluation budget.
    remaining  = max(0,min(maxBudget,Problem.maxFE - Problem.FE));
    if remaining <= 0 || isempty(CV)
        return;
    end
    % Distribute the remaining budget evenly across the variables that need
    % to be analyzed.
    nCor = min(nCor,max(1,floor(remaining/max(1,length(CV))/3)));
    startFE = Problem.FE;
    for v = CV
        RelatedSet = [];
        for d = 1 : length(CVSet)
            for u = CVSet{d}
                drawnow();
                sign = false;
                for i = 1 : nCor
                    if Problem.FE >= Problem.maxFE || Problem.FE - startFE >= maxBudget
                        return;
                    end
                    p    = Population(randi(length(Population)));
                    a2   = unifrnd(Problem.lower(v),Problem.upper(v));
                    b2   = unifrnd(Problem.lower(u),Problem.upper(u));
                    decs = repmat(p.dec,3,1);
                    decs(1,v)     = a2;
                    decs(2,u)     = b2;
                    decs(3,[v,u]) = [a2,b2];
                    F = Problem.Evaluation(decs);
                    delta1 = F(1).obj - p.obj;
                    delta2 = F(3).obj - F(2).obj;
                    if any(delta1.*delta2<0)
                        sign = true;
                        RelatedSet = [RelatedSet,d];
                        break;
                    end
                end
                if sign
                    break;
                end
            end
        end
        if isempty(RelatedSet)
            CVSet = [CVSet,v];
        else
            CVSet = [CVSet,[cell2mat(CVSet(RelatedSet)),v]];
            CVSet(RelatedSet) = [];
        end
    end
end