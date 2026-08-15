# Research

Short numerical walk-throughs of the work this code was written for. One notebook per paper
or thesis chapter, each committed with its output so it reads without being run.

All three use the same structure: a deep circular arch on four design radii, loaded at the
crown and traced with the arc-length solver, over a design box narrower than the paper's.

## `ArcLengthControlledShapeDesign.ipynb`

> Bouwer, J. M., Kok, S. and Wilke, D. N. (2023). *Challenges and solutions to arc-length
> controlled structural shape design problems.* Mechanics Based Design of Structures and
> Machines, **51**(7), 4088–4119.
> [doi:10.1080/15397734.2021.1950549](https://doi.org/10.1080/15397734.2021.1950549)

**Goal.** Recover an arch's shape from its load-deflection path, using the paper's
curve-matching objective and its analytical gradient.

- **Large arc-length steps pay for themselves.** Cost is U-shaped in the prescribed step,
  not monotonic: small steps cost steps, large ones cost retries.
- **The adaptive arc stepping makes the objective discontinuous**, and SLSQP terminates
  *successfully* well short of a known optimum, because it decides termination from
  function values.
- **ADAM gets substantially closer for far fewer solves.** It never reads the objective
  value, which is the paper's argument for defining the optimum by gradient alone.
- **A design the solver cannot trace is usually a starved corrector, not an impossible
  arch.** The iteration budget binds, not the cut budget. Such designs form a thin sheet
  an optimiser can converge onto and then oscillate across, taking a fabricated penalty
  value as it does — so a failed solve is retried once on a larger budget before it counts
  as untraceable, leaving the retry pattern and its discontinuity untouched.

Still to come: the paper's own discontinuity study, and its GOSSA and Modified Subgradient
optimisers.

## `SurrogateBasedShapeDesign.ipynb`

> Bouwer, J., Wilke, D. N. and Kok, S. (2024). *A novel and fully automated coordinate system
> transformation scheme for near optimal surrogate construction.* Computer Methods in Applied
> Mechanics and Engineering, **419**, 116648.
> [doi:10.1016/j.cma.2023.116648](https://doi.org/10.1016/j.cma.2023.116648)

**Goal.** Fit the load path itself — crown load factor and displacement over the design radii
**and** the accumulated arc length — rather than fitting an objective computed from it.

- **One traced path contributes a sample per arc step**, not a single sample.
- **An objective becomes a calculation on top of the fitted responses**, so it can be changed
  or replaced without sampling anything again.
- Fitting over arc length needs the solver's `dLds` and `dUds_All`, which the notebook checks
  against the two identities that make them self-checking before using them.

Needs the optional `ge_rbf` dependency (`uv sync --extra surrogate`); the glue between the
solver's output and its trajectory API is [`fem2d.surrogate`](../src/fem2d/surrogate.py).

## `SurrogateBasedOptimisation.ipynb`

> Bouwer, J. M. (2023). *The shape optimisation of compliant structures to produce a desired
> snap-through load displacement path*, **Chapter 6**. PhD thesis, University of Pretoria.
> [repository.up.ac.za](https://repository.up.ac.za/items/db54b751-f3d3-4387-914a-27c46a2bcc81)

**Goal.** Solve the 2023 paper's design problem on those fitted responses rather than on the
solver. Same box and target as the first notebook, so the objective is the same function and
the two solver-based optimisers serve as baselines.

- **The discontinuity goes.** Evaluating the fitted responses at the target's *fixed* arc
  lengths means neither the sample count nor the sample locations move with the design.
- **It finds a better design than either solver-based optimiser, for fewer solves than
  either** — and the ranking is measured by verification solves, not predicted by the fit.
- **The fit's own error is now what limits how finely it can aim**, rather than the
  optimiser. That is what makes infill the obvious next step, and it is deliberately not
  here.
