function CA = UpdateCA(CA,New,MaxSize,p)
% Update CA

    if isempty(CA)
        CA = INDIVIDUAL.empty();
    end
    if isempty(New)
        return;
    end

    CA = [CA,New];
    N  = length(CA);
    if N <= MaxSize || MaxSize <= 0
        return;
    end

    %% Calculate the fitness of each solution
    [SDE_fitness,~,SDE_Distance] = SDE_plus_indicator(CA.objs,0,p);

    %% Delete part of the solutions by their fitnesses
    if sum(SDE_fitness ~= 0) <= MaxSize
        % One-time evaluation
        [~,Remain] = sortrows(sort(SDE_Distance,2),'descend');
        Remain = Remain(1:min(MaxSize,numel(Remain)));
        if numel(Remain) < MaxSize
            others = setdiff(1:N,Remain,'stable');
            Remain = [Remain,others(1:min(MaxSize-numel(Remain),numel(others)))];
        end
    else
        % Dynamic evaluation
        Choose = SDE_fitness ~= 0;
        while sum(Choose) > MaxSize
            Remain = find(Choose);
            mat    = min(SDE_Distance(Choose,Choose),[],2);
            [~,x]  = min(mat);
            Choose(Remain(x)) = false;
        end
        Remain = find(Choose);
    end
    CA = CA(Remain);
end
