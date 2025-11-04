function [SDE_fitness,SDEP_fitness,Distance] = SDE_plus_indicator(PopObj,f,p)
    N = size(PopObj,1);
    if N == 0
        SDE_fitness  = zeros(0,1);
        SDEP_fitness = zeros(1,0);
        Distance     = zeros(0,0);
        return;
    end
    minObj = min(PopObj,[],1);
    range  = max(PopObj,[],1) - minObj;
    range(range<=0) = 1;
    PopObj = (PopObj - repmat(minObj,N,1))./repmat(range,N,1);
    PopObj(~isfinite(PopObj)) = 0;
    Distance = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        diff    = repmat(PopObj(i,:),N,1) - SPopObj;
        diff(i,:) = 0;
        Distance(i,:) = (sum(abs(diff).^p,2).^(1/p))';
        Distance(i,i) = inf;
    end
    SDEP_fitness = zeros(1,N);
    if f == 1
        [~,Remain] = sortrows(sort(Distance,2),'descend');
        SDEP_fitness(Remain) = N:-1:1;
    end
    SDE_fitness = min(Distance,[],2);
end
