function DA = UpdateDA_VaEA(~, Union, N_DA, zmin, zmax, nSeed)
% 多样性优先（VaEA思想）：
% 先取极值种子，再在归一化目标空间用“最大最小夹角优先”贪心填充

    if isempty(Union)
        DA = Union; return;
    end

    [ObjsN, ~, ~] = VaEAUtils.NormalizeObjs(Union.objs, zmin, zmax);
    seedIdx = VaEAUtils.ExtremeSeeds(ObjsN, min(nSeed, N_DA));
    selIdx  = VaEAUtils.AngleGreedy(ObjsN, N_DA, seedIdx);
    DA      = Union(selIdx);
end
