function CrowdDis = ConvergenceScore(front,p)

    [m,~] = size(front);
    CrowdDis = zeros(1,m);

    for i = 1 : m
        value = norm(front(i,:),p);
        if value > 0
            CrowdDis(i) = 1./value;
        else
            CrowdDis(i) = inf;
        end
    end
end
