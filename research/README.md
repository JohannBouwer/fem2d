# Research

Numerical demonstrations behind

> Bouwer, J. M., Kok, S. and Wilke, D. N. (2023). *Challenges and solutions to arc-length
> controlled structural shape design problems.* Mechanics Based Design of Structures and
> Machines, **51**(7), 4088–4119.
> [doi:10.1080/15397734.2021.1950549](https://doi.org/10.1080/15397734.2021.1950549)

**[`ArcLengthControlledShapeDesign.ipynb`](ArcLengthControlledShapeDesign.ipynb)** — takes
about six minutes to run, and is committed with its output so it reads without running.

## The structure

A deep circular arch spanning 215°, clamped at one end and pinned at the other, loaded at
the crown. The shape parameters are the arch radius at four equally spaced angles, with
the two ends held at the base radius and a cubic spline through all six points.

The paper publishes the geometry and design setup — radius 100 mm, design radii between 80
and 120 mm, ends held at 100 mm — but not the material constants. The section is therefore
taken from the DaDeppo and Schmidt benchmark the arch comes from, stated as
`EA = EI = 1e6`. For a rectangular section of unit out-of-plane thickness that fixes both
the depth and the modulus: `h = sqrt(12)` and `E = 1e6/sqrt(12)`, plane stress.

## 1. Large arc-length steps pay for themselves

Prescribed arc length swept on one fixed design, 5β elements, 20 elements, total arc
length 800:

| arc length | steps | arc cuts | time (s) | λ max |
|---|---|---|---|---|
| 160 | 12 | 39 | 48.1 | 1246.0 |
| 80 | 15 | 23 | 28.7 | 1242.1 |
| **40** | 23 | 9 | **23.4** | 1241.7 |
| 20 | 41 | 2 | 33.0 | 1247.1 |
| 10 | 80 | 0 | 54.3 | 1247.4 |

Times are wall clock from one run and will vary; the step and cut counts will not.

λ max moves by 0.4% across the sweep, so every run traced the same path and the times are
comparable.

The paper's Table 1 reports the same trend monotonically on its own arch — 26.2, 42.5,
53.7, 89.6 and 163.8 seconds for step lengths of 100, 50, 25, 12.5 and 6.25 — so halving
the step roughly doubles the cost.

What this sweep adds is the other end of the curve. Past a point a larger step stops
paying, because the solver spends its time cutting back and retrying: 39 reductions at a
step of 160 against 9 at 40. **The cost is U-shaped, not monotonic**, and the cheapest
setting is the largest step that is not provoking many retries.

## 2 and 3. A four-variable design space, solved with Q8

Four designs, each traced with the arc-length solver on eight-node elements:

| design | radii | steps | time (s) | limit load |
|---|---|---|---|---|
| uniform | 100, 100, 100, 100 | 20 | 44.7 | 2345.4 |
| raised | 88, 112, 112, 88 | 19 | 39.8 | 2267.7 |
| flattened | 112, 88, 88, 112 | 19 | 48.0 | 1754.7 |
| skewed | 92, 118, 96, 108 | 18 | 37.9 | 1792.4 |

The flattened design snaps *back*: its path reverses in displacement as well as in load.
Neither load control nor displacement control can trace a curve like that, which is why
the design study needs arc-length control at all.

## Caveat on the limit load

The peak load factor is **not mesh converged** for this problem as set up. Refining along
the arch with everything else fixed gives, for the uniform design with 5β elements, limit
loads of 1247, 783, 672 and 616 at 20, 40, 60 and 80 elements, with the crown deflection
at the limit falling alongside it.

It is not a point-load singularity: sharing the load across the crown cross-section moves
the answer by under 0.1%. It is not the solver either, since refining the arc length on a
*fixed* mesh converges cleanly, 783.26 to 783.23. The likely cause is that the arc-length
constraint sums over every free degree of freedom, so `||dU||` grows roughly as
`sqrt(ndof)` and a fixed total arc length covers less of the physical path as the mesh is
refined. **This is open.**

Nothing above depends on it — the sweep holds the mesh fixed and varies only the step, and
the design comparison holds the mesh fixed and varies only the shape — but the limit loads
should be read relative to one another, not as converged values.

## Not covered yet

The optimisation half of the paper: the curve-matching objective functions A and B, the
discontinuities that adaptive arc stepping creates in them, and the gradient-only
optimisers (GOSSA, Modified Subgradient) that tolerate those discontinuities where SLSQP
mistakes a discontinuity for a minimum.
