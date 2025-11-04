function [DA,p] = UpdateDA(DA,New,MaxSize)
% Update DA

    if isempty(DA)
        DA = INDIVIDUAL.empty();
    end
    if isempty(New)
        if isempty(DA)
            p = 1;
        else
            front1 = DA.objs;
            IdealPoint = min(front1,[],1);
            [~,p] = SurvivalScore(front1,IdealPoint);
        end
        return;
    end

    DA = [DA,New];
    ND = NDSort(DA.objs,1);
    DA = DA(ND==1);
    N  = length(DA);
    front1 = DA.objs;
    if isempty(front1)
        p = 1;
        return;
    end
    IdealPoint = min(front1,[],1);
    [CrowdDis,p] = SurvivalScore(front1,IdealPoint);
    if N <= MaxSize
        return;
    end

    [~,Rank] = sort(CrowdDis,'descend');
    DA = DA(Rank(1:min(MaxSize,N)));
end
