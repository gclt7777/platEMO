function divValue = CalculateDiv_Test(PopObj,q)
% 多样性计算函数

    [N,M] = size(PopObj);
    if N < 2
        divValue = zeros(1,N);
        return;
    end

    Distance = zeros(N,N);
    divValue = zeros(1,N);
    limit = max(min(q, N-1), 1);

    for i = 1 : N
        for j = 1 : N
            diff = PopObj(i,:) - PopObj(j,:);
            Distance(i,j) = sqrt(sum(diff.^2) - sum(diff)^2/M);
        end
        sortDis = sort(Distance(i,:), 'ascend');
        for j = 1 : limit
            divValue(i) = divValue(i) + exp(-(j-1))*sortDis(j+1);
        end
    end
end
