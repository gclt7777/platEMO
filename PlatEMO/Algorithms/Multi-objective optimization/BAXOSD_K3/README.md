# BAXOSD\_K3 notes

## Algorithm summary
- **Bi-axis exploration** divides each generation into a convergence-oriented C-phase and a diversity-oriented D-phase, balanced via `theta = (FE/maxFE)^alpha`.
- **Objective grouping (O)** clusters objectives by angular distance; each group receives decision variable subsets (S) designed from the inverse mapping energy to balance A/P variable quotas and keep every group populated.
- **C-phase** selects valid O–S pairs, performs PCA on grouped objectives, evolves in the latent space with DE/rand/1/bin, and back-projects only onto the group-specific decision columns using a half-step fusion factor `eta`.
- **D-phase** associates solutions with scaled reference vectors, runs DE in objective space within each sector, and writes the full-column decision increments via the linear inverse map before half-step fusion and boundary repair. Parent sampling now uses PlatEMO's tournament selector with feasibility-weighted angular penalties to stay consistent with platform constraint handling.
- **PlatEMO interoperability:** a matrix-view helper (`pop_matrix_view`) reads `INDIVIDUAL` arrays into dense decision/objective matrices (plus feasibility flags) for vectorized math, while offspring creation stays routed through `Problem.Evaluation` and a shared bound clamp so boundary treatment remains consistent with PlatEMO tools.
- **Inverse mapping** fits a z-score linear decoder `T` each generation via ridge-regularized SVD, limiting condition number by `kappa_tar` and storing scaling statistics for Δy→Δx projection.
- **Environmental selection** follows RVEA-APD with feasibility-first handling: one survivor per sector using APD, then fills remaining slots prioritizing C-phase then D-phase offspring.

## Potential optimizations & consistency improvements
- **Reference vector updates:** expose the `fr` frequency as a user parameter or auto-tune based on objective drift to avoid stale vector orientations on highly non-stationary problems.
- **Grouping stability:** cache `O_groups` and `S_groups` along with their statistics; when periodic regrouping is skipped, reuse previous PCA bases or inertia weights to reduce per-generation overhead on large M.
- **Vectorization:** replace per-offspring loops in C/D phases with batched sampling where feasible (e.g., pre-generating parent indices and PCA bases per group) to better align with PlatEMO vectorized operators.
- **Constraint handling:** pass constraint violations into the D-phase sector selection to bias sampling toward feasible sectors, mirroring RVEA’s feasibility-first philosophy during variation as well as selection.
- **Test coverage:** validate the operator on LSMOP, DTLZ, and WFG suites with varying objective counts to ensure `rk_cap_minmax` and `rhoA` keep groups non-empty and the half-step fusion stays stable under different decision bounds.

## Usage on PlatEMO
- The algorithm exposes a single public parameter `alpha` (default `1.6`), which controls the steepness of the progress scheduler `theta = (FE/maxFE)^alpha` that balances C- and D-phase effort.
- Default internal settings follow the source (e.g., `rhoA = 0.5`, `periodGroup = 10`, `EVR_target = 0.85`, `eta = 0.5`, and `fr = inf` for a fixed reference set). Tune `fr` and `periodGroup` together when objectives drift quickly.
- Typical PlatEMO invocation for benchmark suites:
  ```matlab
  % DTLZ / WFG
  platemo('algorithm',@BAXOSD_K3,'problem',@DTLZ2,'M',5,'D',12,'maxFE',100000,'alpha',1.6);
  platemo('algorithm',@BAXOSD_K3,'problem',@WFG3,'M',3,'D',24,'maxFE',50000);

  % LSMOP (large-scale)
  platemo('algorithm',@BAXOSD_K3,'problem',@LSMOP4,'M',3,'D',1000,'N',120,'maxFE',200000);
  ```
- The code respects the user-specified population size `N`. For many-objective WFG/DTLZ settings, start with the default reference vectors by omitting `N`; for LSMOP, increasing `N` to 120–200 can stabilize sector coverage.
- Environmental selection is RVEA-APD with feasibility-first handling. If an experiment needs stronger pressure on feasibility, adjust constraint treatment inside `Operator_DPhase` and `EnvironmentalSelection_BAXOSD` following the constraints hook noted above.

