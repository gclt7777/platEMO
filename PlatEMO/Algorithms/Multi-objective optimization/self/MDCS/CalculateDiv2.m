function Distance = CalculateDiv2(PopObj,N)
% Calculate parallel distance

    if N == 0
        Distance = [];
        return;
    end

    [~,M] = size(PopObj);
    if N == 1
        Distance = 0;
        return;
    end

    range = max(max(PopObj)-min(PopObj),1e-12);
    PopObj = (PopObj-repmat(min(PopObj),N,1))./repmat(range,N,1);

    Distance = zeros(N,N);
    for i = 1 : N
        Fi     = PopObj(i,:);
        Fdelta = PopObj - repmat(Fi,N,1);
        Distance(i,:) = sqrt(sum(Fdelta.^2,2)-(sum(Fdelta,2)).^2./M);
        Distance(i,i) = Inf;
    end
end
