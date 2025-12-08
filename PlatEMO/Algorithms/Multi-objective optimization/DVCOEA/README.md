# DVCOEA in PlatEMO

This folder contains the MATLAB implementation of the **Dimensionally-Varying Coevolutionary Algorithm (DVCOEA)** for multi-objective optimization. The code detects convergence-related and diversity-related variables, optimizes them separately, and keeps the population diverse.

## Main workflow
- `DVCOEA.m` wires the algorithm together. It sets parameters `nSel`, `nPer`, and `nCor` (defaults: 5, 50, 5), initializes the archive, clusters variables, performs correlation analysis, and then loops until termination with separate convergence and distribution optimization phases.
- `VariableClustering.m` perturbs decision variables to identify convergence-related variables (CV), diversity-related variables (DV), and the objectives each CV contributes to (`CO`).
- `CorrelationAnalysis.m` groups convergence-related variables that are highly correlated.
- `ConvergenceOptimization.m` evolves the convergence-related groups using differential evolution on the selected indices.
- `DistributionOptimization.m` promotes population diversity by evolving diversity-related variables and any correlated convergence variables before environmental selection.

## Usage notes
1. Select **DVCOEA** from the algorithm list when setting up experiments in PlatEMO.
2. Adjust `nSel`, `nPer`, and `nCor` via the algorithm parameters if different sampling or correlation sensitivity is desired.
3. The algorithm assumes the problem definition provides appropriate bounds for all decision variables.

## 中文简介
- DVCOEA通过扰动和聚类将决策变量划分为收敛相关变量（CV）和多样性相关变量（DV），并记录每个CV影响的目标集合（`CO`）。
- 相关性分析会把高度相关的收敛变量合并，再分别用差分进化优化收敛组、用分布优化保持多样性。
- 可在算法参数中调整`nSel`（扰动样本数）、`nPer`（扰动幅度）、`nCor`（相关性阈值）以适配不同问题特性。

These files correspond to the DVCOEA implementation originally hosted at [BIMK/PlatEMO](https://github.com/BIMK/PlatEMO/tree/master/PlatEMO/Algorithms/Multi-objective%20optimization/DVCOEA).
