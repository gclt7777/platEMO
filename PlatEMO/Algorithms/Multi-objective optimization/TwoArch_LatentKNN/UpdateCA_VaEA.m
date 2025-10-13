function CA = UpdateCA_VaEA(~, Union, N_CA, zmin, zmax)
% 收敛优先：
% 1) 非支配排序逐层填满
% 2) 临界层按 “fit = sum(归一化目标)” 升序截断

    if isempty(Union)
        CA = Union; return;
    end

    [FrontNo, MaxFNo] = NDSort(Union.objs, N_CA);
    [ObjsN, ~, ~] = VaEAUtils.NormalizeObjs(Union.objs, zmin, zmax);
    [scoreAll, ~, distAll] = VaEAUtils.ConvergenceScore(ObjsN);
    keep = FrontNo < MaxFNo;
    CA   = Union(keep);

    remain = N_CA - length(CA);
    if remain > 0
        lastIdx = find(FrontNo == MaxFNo);
        metrics = [scoreAll(lastIdx), distAll(lastIdx)];
        [~, order] = sortrows(metrics, [1 2]);
        take = min(remain, numel(order));
        CA = [CA, Union(lastIdx(order(1:take)))];
    elseif length(CA) > N_CA
        [ObjsN_CA, ~, ~] = VaEAUtils.NormalizeObjs(CA.objs, zmin, zmax);
        [scoreCA, ~, distCA] = VaEAUtils.ConvergenceScore(ObjsN_CA);
        metrics = [scoreCA, distCA];
        [~, ord] = sortrows(metrics, [1 2]);
        CA = CA(ord(1:min(N_CA, numel(ord))));
    end
end
