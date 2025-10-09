# OS\_DNS\_Soft 收敛性分析

- **线性预像假设过于强烈**：算法使用 `learnT_linear` 通过岭回归学习从目标空间到决策空间的线性映射（`X \approx zscore(Y) * T`）。当真实的目标-决策映射高度非线性或存在多模态时，线性模型会产生较大偏差，导致 `mapY2X_linear` 输出的候选解远离帕累托前沿，进而拖慢收敛速度。【F:PlatEMO/Algorithms/Multi-objective optimization/OS_DNS_Soft/OS_DNS_Soft.m†L45-L72】
- **全体松弛更新的跟随步长敏感**：在 `Y_new = (1-α)Y + α[(1-β)Y* + β z_min]` 中，较大的 `α` 或 `β` 会使群体快速向软教师或理想点靠拢。一旦 `Y*` 质量不佳或 `z_min` 偏移严重，整体会被牵引到欠优区域，造成震荡或停滞。【F:PlatEMO/Algorithms/Multi-objective optimization/OS_DNS_Soft/OS_DNS_Soft.m†L38-L49】
- **缺乏多样化算子**：算法几乎完全依赖软教师更新和线性映射，没有显式交叉/变异来探索决策空间。当 `Y*` 供给的信息不足时，搜索会快速收缩，多样性下降，也会表现为收敛缓慢或早熟停滞。【F:PlatEMO/Algorithms/Multi-objective optimization/OS_DNS_Soft/OS_DNS_Soft.m†L34-L66】

因此，当上述条件叠加（问题强非线性、参数设置偏激、软教师质量不稳）时，就容易观察到 OS\_DNS\_Soft 的收敛性较差。
