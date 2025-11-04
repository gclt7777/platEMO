function [CrowdDis,p,normalization] = SurvivalScore(front,IdealPoint)

    [m,n] = size(front);
    CrowdDis = zeros(1,m);

    if m == 0
        p = 1;
        normalization = [];
        return;
    end
    if m < n
        p = 1;
        normalization = max(front,[],1)';
        return;
    end

    % shift the ideal point to the origin
    front = front - IdealPoint;

    % Detect the extreme points and normalize the front
    Extreme = FindCornerSolutions(front);
    [front,normalization] = Normalize(front,Extreme);
    front(~isfinite(front)) = 0;

    % set the distance for the extreme solutions
    Extreme = unique(Extreme(Extreme>=1 & Extreme<=m));
    CrowdDis(Extreme) = inf;
    selected = false(1,m);
    selected(Extreme) = true;

    % approximate p (norm)
    d = Point2LineDistance(front,zeros(1,n),ones(1,n));
    d(Extreme) = inf;
    [~,index] = min(d);
    meanFront = mean(front(index,:));
    if meanFront > 0 && meanFront ~= 1
        p = log(n)/log(1/meanFront);
    else
        p = 1;
    end
    if isnan(p) || ~isreal(p) || p <= 0.1
        p = 1;
    end

    if ~any(selected)
        selected(index) = true;
        CrowdDis(index) = inf;
    end

    nn = sum(abs(front).^p,2).^(1/p);
    nn(nn<=0) = 1;
    distances = inf(m);
    for i = 1 : m
        SPopObj = max(front,repmat(front(i,:),m,1));
        diff    = front - SPopObj;
        diff(i,:) = 0;
        distances(i,:) = (sum(abs(diff).^p,2).^(1/p))';
        distances(i,i) = inf;
    end

    distances = distances./repmat(nn,1,m);

    remaining = find(~selected);
    baseSelected = sum(selected);
    for i = 1 : m - baseSelected - 1
        if isempty(remaining)
            break;
        end
        currentNeighbors = min(2,sum(selected));
        if currentNeighbors <= 0
            currentNeighbors = 1;
        end
        maxim = mink(distances(remaining,selected),currentNeighbors,2);
        dscore = sum(maxim,2);
        CrowdDis1 = ConvergenceScore(front(remaining,:),p);
        [c,index] = max(dscore.*CrowdDis1');
        best = remaining(index);
        remaining(index) = [];
        selected(best) = true;
        CrowdDis(best) = c;
    end
end

function [front,normalization] = Normalize(front,Extreme)
    [m,n] = size(front);

    if length(Extreme) ~= length(unique(Extreme))
        normalization = max(front,[],1)';
        normalization(normalization<=0) = 1;
        front = front./repmat(normalization',m,1);
        return
    end

    % Calculate the intercepts of the hyperplane constructed by the extreme
    % points and the axes
    Hyperplane = front(Extreme,:)\ones(n,1);
    if any(isnan(Hyperplane)) || any(isinf(Hyperplane)) || any(Hyperplane<0)
         normalization = max(front,[],1)';
    else
        normalization = 1./Hyperplane;
        if any(isnan(normalization)) || any(isinf(normalization))
            normalization = max(front,[],1)';
        end
    end
    % Normalization
    normalization(normalization<=0) = 1;
    front = front./repmat(normalization',m,1);
end
