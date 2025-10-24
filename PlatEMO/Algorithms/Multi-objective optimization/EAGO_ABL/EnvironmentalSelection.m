function Population = EnvironmentalSelection(Population,N)
% The environmental selection of distribution optimization in LMEA / EAGO

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = FrontNo < MaxFNo;

    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = Truncation(Population(Last).objs, N - sum(Next));
    Next(Last(Choose)) = true;

    %% Population for next generation
    Population = Population(Next);
end

function Choose = Truncation(PopObj,K)
% Select part of the solutions by truncation (angle-based)

    %% Normalize and compute pairwise cosine
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    span   = fmax - fmin;
    span(span==0) = 1;                             % 数值稳健
    PopObj = (PopObj - fmin) ./ span;

    Cosine = 1 - pdist2(PopObj,PopObj,'cosine');
    Cosine(1:size(Cosine,1)+1:end) = 0;            % diag = 0

    %% Truncation：先留极端点，再按最远离已选的准则选
    Choose = false(1,size(PopObj,1));
    [~,extreme] = max(PopObj,[],1);                % 每个目标的极端点
    Choose(extreme) = true;

    if sum(Choose) > K
        selected = find(Choose);
        Choose   = selected(randperm(length(selected),K));
    else
        while sum(Choose) < K
            unSelected = find(~Choose);
            [~,x]      = min(max(Cosine(~Choose,Choose),[],2));
            Choose(unSelected(x)) = true;
        end
    end
end
