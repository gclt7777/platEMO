function DA = UpdateDA(DA,New,MaxSize,W)
% Update DA

    %% Find the non-dominated solutions
    DA = [DA,New];
    ND = NDSort(DA.objs,1);
    DA = DA(ND==1);  % 保留非支配解
    N  = length(DA);
    if N <= MaxSize
        return;
    end
    Popobj = DA.objs; % 第一个非支配层级保留的解

    %% 归一化
    Zmin   = min(Popobj,[],1);
    range  = max(max(Popobj,[],1)-Zmin,1e-12);
    Popobj = (Popobj-repmat(Zmin,size(Popobj,1),1))./repmat(range,size(Popobj,1),1);
    Choose = false(1,N);

    %% 关联参考向量
    NZ       = size(W,1);
    Cosine   = 1 - pdist2(Popobj,W,'cosine');
    Distance = repmat(sqrt(sum(Popobj.^2,2)),1,NZ).*sqrt(1-Cosine.^2);
    % 每个解对应的参考向量 pi 及其与参考向量之间的距离 d
    [~,pi] = min(Distance',[],1); %#ok<ASGLU>
    % 获取每个参考向量关联的个体
    [~,index] = min(Distance,[],1);

    %% 选择极值点
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
        [~,localIdx]   = min(penaltyMeasure); % 带惩罚的 AFS 函数值
        Extreme(i)     = remaining(localIdx);
        Choose(Extreme(i)) = true;
    end

    % 遍历每个参考向量
    for i = 1 : NZ
        Choose(index(i)) = true;
    end

    if sum(Choose) > MaxSize
        % 随机删除一些解
        Choosed = find(Choose);
        dropIdx = randperm(numel(Choosed),numel(Choosed)-MaxSize);
        Choose(Choosed(dropIdx)) = false;
    elseif sum(Choose) < MaxSize
        %% 计算种群信息熵
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
        UnselectedDiv  = divValue(Unselected);
        [~,indexDiv]   = sort(UnselectedDiv,'descend');
        Choose(Unselected(indexDiv(1:need))) = true;
    end

    DA = DA(Choose);
end
