function Offspring_dec = Mutation(localgroup,i,k,Parent,lower,upper,mutationStrength,Population,L1list)
% Mutation operator used in LSMaODE.

    ParentDec    = Parent.dec;
    Offspring_dec = ParentDec;

    if i <= localgroup
        % Local subpopulation evolution
        if rand > 0.1
            dim = k;
            Offspring_dec(dim) = gaussianMutate(ParentDec(dim),lower(dim),upper(dim),mutationStrength);
        else
            temp_localPop = Population;
            if ~isempty(temp_localPop) && i <= length(temp_localPop)
                temp_localPop(i) = [];
            end
            if length(temp_localPop) < 3
                dim = k;
                Offspring_dec(dim) = gaussianMutate(ParentDec(dim),lower(dim),upper(dim),mutationStrength);
                return;
            end
            idx     = randperm(length(temp_localPop),3);
            Parent1 = temp_localPop(idx(1)).dec;
            Parent2 = temp_localPop(idx(2)).dec;
            Parent3 = temp_localPop(idx(3)).dec;
            if rand > 0.5
                bounds   = [Parent1;Parent2];
                tempDec  = Parent1 + rand.*(Parent2-Parent1);
            else
                bounds   = [Parent1;Parent2;Parent3];
                tempDec  = Parent1 + rand.*(Parent2-Parent3);
            end
            dim = k;
            lowerBound = min(bounds(:,dim));
            upperBound = max(bounds(:,dim));
            Offspring_dec(dim) = min(max(lowerBound,tempDec(dim)),upperBound);
        end
    else
        % Global population evolution
        L1len = numel(L1list);
        if rand > 0.5 && L1len >= 3
            idx     = randperm(L1len,3);
            Parent1 = Population(L1list(idx(1))).dec;
            Parent2 = Population(L1list(idx(2))).dec;
            Parent3 = Population(L1list(idx(3))).dec;
            if rand > 0.5
                bounds        = [Parent1;Parent2];
                Offspring_dec = Parent1 + rand.*(Parent2-Parent1);
            else
                bounds        = [Parent1;Parent2;Parent3];
                Offspring_dec = Parent1 + rand.*(Parent2-Parent3);
            end
            lowerBound = min(bounds,[],1);
            upperBound = max(bounds,[],1);
            Offspring_dec = min(max(lowerBound,Offspring_dec),upperBound);
        else
            dim = randi(length(Offspring_dec));
            Offspring_dec(dim) = gaussianMutate(ParentDec(dim),lower(dim),upper(dim),mutationStrength);
        end
    end
end

function value = gaussianMutate(current,lower,upper,mutationStrength)
% Perform a Gaussian perturbation constrained within the variable bounds.

    sigma = (upper - lower)/mutationStrength;
    if sigma <= 0
        value = min(max(lower,current),upper);
        return;
    end
    for attempt = 1 : 50
        candidate = current + sigma*randn;
        if candidate >= lower && candidate <= upper
            value = candidate;
            return;
        end
    end
    value = min(max(lower,current),upper);
end
