function DA = UpdateDA_tree(DA, Union, MaxSize, zmin, zmax, nSeed)
% DA（多样性优先）：极值种子 + FPS（在归一化目标空间）

    if isempty(Union)
        DA = SOLUTION.empty(); return;
    end

    Pop = Union;
    Y   = Pop.objs;
    Yn  = TreeUtils.NormalizeObjs(Y, zmin, zmax);
    N   = length(Pop);
    M   = size(Y,2);

    % 极值种子
    [~,idxMin] = min(Yn,[],1);
    [~,idxMax] = max(Yn,[],1);
    seeds = unique([idxMin(:); idxMax(:)].','stable');
    if nargin<6 || isempty(nSeed), nSeed = M; end
    if numel(seeds) > nSeed
        seeds = seeds(1:nSeed);
    end
    if isempty(seeds)
        seeds = randi(N);
    end

    % FPS 填满
    pick = FPS_fill(Yn, seeds, MaxSize);
    DA = Pop(pick);
end

function pick = FPS_fill(Yn, seedIdx, K)
    N = size(Yn,1);
    pick = unique(seedIdx(:).');
    if isempty(pick), pick = randi(N); end
    dist = min(pdist2_safe(Yn, Yn(pick,:)),[],2);
    dist(pick) = 0;
    while numel(pick) < K
        [~,idx] = max(dist);
        pick(end+1) = idx; %#ok<AGROW>
        dist = min(dist, pdist2_safe(Yn, Yn(idx,:)));
        dist(pick) = 0;
    end
end

function D = pdist2_safe(A,B)
    try
        D = pdist2(A,B);
    catch
        AA = sum(A.^2,2); BB = sum(B.^2,2).';
        D = sqrt(max(0, bsxfun(@plus,AA,BB) - 2*(A*B.')));
    end
end
