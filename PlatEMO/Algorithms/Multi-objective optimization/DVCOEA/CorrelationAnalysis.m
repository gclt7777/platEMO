function CVSet = CorrelationAnalysis(Global,Population,CV,nCor)
% Detect the group of each distance variable  
    CVSet = {};
    for v = CV
        RelatedSet = [];
        for d = 1 : length(CVSet)
            for u = CVSet{d}
                drawnow();
                sign = false;
                for i = 1 : nCor
                    p    = Population(randi(length(Population)));
                    a2   = unifrnd(Global.lower(v),Global.upper(v));
                    b2   = unifrnd(Global.lower(u),Global.upper(u));
                    decs = repmat(p.dec,3,1);
                    decs(1,v)     = a2;
                    decs(2,u)     = b2;
                    decs(3,[v,u]) = [a2,b2];
                    F = INDIVIDUAL(decs);
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