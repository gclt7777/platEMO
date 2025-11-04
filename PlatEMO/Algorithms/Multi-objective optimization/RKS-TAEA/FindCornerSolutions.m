function indexes = FindCornerSolutions(front)

    [m,n] = size(front);
    if m == 0
        indexes = [];
        return;
    end

    % let's normalize the objectives
    if m <= n
        indexes = 1:m;
        return;
    end

    % let's define the axes of the n-dimensional spaces
    W = eye(n) + 1e-6;
    indexes = zeros(1,n);
    for i = 1 : n
        [~,index] = min(Point2LineDistance(front,zeros(1,n),W(i,:)));
        indexes(i) = index;
    end
    indexes = unique(indexes,'stable');
end
