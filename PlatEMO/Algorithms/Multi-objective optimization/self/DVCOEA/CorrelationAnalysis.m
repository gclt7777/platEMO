function CVSet = CorrelationAnalysis(Problem,Population,CV,nCor)
% Detect the group of each convergence-related variable

    CVSet = cell(0);
    for v = CV(:)'
        related = [];
        for d = 1 : length(CVSet)
            group = CVSet{d};
            sign  = false;
            for u = group
                for i = 1 : nCor
                    p    = Population(randi(length(Population)));
                    a2   = unifrnd(Problem.lower(v),Problem.upper(v));
                    b2   = unifrnd(Problem.lower(u),Problem.upper(u));
                    decs = repmat(p.dec,3,1);
                    decs(1,v)     = a2;
                    decs(2,u)     = b2;
                    decs(3,[v,u]) = [a2,b2];
                    F = Problem.Evaluation(decs);
                    delta1 = F(1).objs - p.objs;
                    delta2 = F(3).objs - F(2).objs;
                    if any(delta1.*delta2<0)
                        sign = true;
                        related = [related,d]; %#ok<AGROW>
                        break;
                    end
                end
                if sign
                    break;
                end
            end
        end
        if isempty(related)
            CVSet{end+1} = v; %#ok<AGROW>
        else
            merged = unique([CVSet{related},v]);
            CVSet(related) = [];
            CVSet{end+1} = merged; %#ok<AGROW>
        end
    end
end
