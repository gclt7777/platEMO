function CA = UpdateCA_VaEA(~, Union, N_CA, zmin, zmax)
% 收敛优先：
% 1) 非支配排序逐层填满
% 2) 临界层按 “fit = sum(归一化目标)” 升序截断

    if isempty(Union)
        CA = Union; return;
    end

    [FrontNo, MaxFNo] = NDSort(Union.objs, N_CA);
    keep = FrontNo < MaxFNo;
    CA   = Union(keep);

    remain = N_CA - length(CA);
    if remain > 0
        lastIdx = find(FrontNo == MaxFNo);
        [ObjsN, ~, ~] = VaEAUtils.NormalizeObjs(Union.objs, zmin, zmax);
        fit = VaEAUtils.FitScore(ObjsN);
        [~, order] = sort(fit(lastIdx), 'ascend');
        CA = [CA, Union(lastIdx(order(1:remain)))];
    elseif length(CA) > N_CA
        [ObjsN, ~, ~] = VaEAUtils.NormalizeObjs(CA.objs, zmin, zmax);
        fit = VaEAUtils.FitScore(ObjsN);
        [~, ord] = sort(fit, 'ascend');
        CA = CA(ord(1:N_CA));
    end
end
