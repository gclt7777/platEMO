function DA = UpdateDA(DA,New,MaxSize,W)
% Update the dominance archive.

    %% Find the non-dominated solutions
    DA = [DA,New];
    if isempty(DA)
        return;
    end
    ND = NDSort(DA.objs,1);
    DA = DA(ND==1);
    N  = length(DA);
    if N <= MaxSize
        return;
    end
    Popobj = DA.objs;

    %% Normalization
    Zmin   = min(Popobj,[],1);
    range  = max(max(Popobj,[],1)-Zmin,1e-12);
    Popobj = (Popobj-repmat(Zmin,size(Popobj,1),1))./repmat(range,size(Popobj,1),1);
    Choose = false(1,N);

    %% Associate with reference vectors
    NZ       = size(W,1);
    Cosine   = 1 - pdist2(Popobj,W,'cosine');
    Cosine   = min(max(Cosine,-1),1);
    Distance = repmat(sqrt(sum(Popobj.^2,2)),1,NZ).*sqrt(max(0,1-Cosine.^2));
    [~,pi]   = min(Distance',[],1);
    [~,index] = min(Distance,[],1);

    %% Select extreme points
    M       = size(Popobj,2);
    Extreme = zeros(1,M);
    w       = zeros(M)+1e-6+eye(M);
    for i = 1 : M
        remaining = find(~Choose);
        if isempty(remaining)
            break;
        end
        subsetObjs     = DA(remaining).objs;
        penaltyMeasure = max(subsetObjs./repmat(w(i,:),numel(remaining),1),[],2) + 0.1*subsetObjs(:,i)/(1e-6);
        [~,localIdx]   = min(penaltyMeasure);
        Extreme(i)     = remaining(localIdx);
        Choose(Extreme(i)) = true;
    end

    for i = 1 : NZ
        Choose(index(i)) = true;
    end

    if sum(Choose) > MaxSize
        Choosed = find(Choose);
        dropIdx = randperm(numel(Choosed),sum(Choose)-MaxSize);
        Choose(Choosed(dropIdx)) = false;
    elseif sum(Choose) < MaxSize
        UQpi        = unique(pi);
        ConnectZNum = length(UQpi);
        EntropyQ    = CalculateEntropy(UQpi,ConnectZNum,pi,N);
        q = floor(N/ConnectZNum*(1-EntropyQ));
        if q < 1
            q = 1;
        end
        need = MaxSize-sum(Choose);
        divValue       = CalculateDiv_Test(Popobj,q);
        Unselected     = find(~Choose);
        need           = min(need,numel(Unselected));
        if need > 0
            UnselectedDiv  = divValue(Unselected);
            [~,indexDiv]   = sort(UnselectedDiv,'descend');
            Choose(Unselected(indexDiv(1:need))) = true;
        end
    end

    DA = DA(Choose);
end