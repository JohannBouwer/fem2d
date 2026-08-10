# Research

Numerical demonstrations behind

> Bouwer, J. M., Kok, S. and Wilke, D. N. (2023). *Challenges and solutions to arc-length
> controlled structural shape design problems.* Mechanics Based Design of Structures and
> Machines, **51**(7), 4088–4119.
> [doi:10.1080/15397734.2021.1950549](https://doi.org/10.1080/15397734.2021.1950549)

**[`ArcLengthControlledShapeDesign.ipynb`](ArcLengthControlledShapeDesign.ipynb)** — takes
about three minutes to run, and is committed with its output so it reads without running.

## The structure

The paper's arch: a deep circular arch spanning 270°, from −45° to 225°, clamped at one end
and pinned at the other, loaded at the crown. The shape parameters are the arch radius at
four equally spaced angles, with the two ends held at the base radius and a cubic spline
through all six points.

**The section is a choice, not the paper's.** The paper publishes the geometry and design
setup — radius 100 mm, design radii between 80 and 120 mm, ends held at 100 mm — but not the
material constants. Here the radial depth is set so `I/A = h²/12 = 1`, hence `h = sqrt(12)`
and `EA = EI`, with `E = 210` GPa, unit out-of-plane thickness, plane stress.

That only sets the scale of the load axis. With a single elastic modulus the residual is
`R(u) = E·f(u) − λF` with `f` independent of `E`, so `λ` is **linear in `E`**; and with
`ψ = 0` the arc-length constraint is purely `‖ΔU‖`, so the solver visits the same
displacement states whatever `E` is. Verified: changing `E` from 288.7 to 210 GPa left the
displacement path identical to 4e-12 and scaled every load factor by exactly 0.727461. If
the paper's peak load factor is known, `E` can be set to reproduce it, and nothing else
moves.

## 1. Large arc-length steps pay for themselves

Prescribed arc length swept on one fixed design, 5β elements, 20 elements, total arc
length 800:

| arc length | steps | arc cuts | time (s) | λ max |
|---|---|---|---|---|
| 640 | 7 | 34 | 13.2 | 235.7 |
| 320 | 7 | 20 | 9.0 | 235.7 |
| 160 | 7 | 6 | 3.6 | 235.7 |
| **80** | 10 | 0 | **3.0** | 236.9 |
| 40 | 21 | 3 | 4.3 | 237.2 |
| 20 | 41 | 2 | 6.3 | 237.4 |
| 10 | 80 | 0 | 11.2 | 237.4 |

Times are wall clock from one run and will vary; the step and cut counts will not.

λ max moves by 0.7% across the sweep, so every run traced the same path and the times are
comparable.

The paper's Table 1 reports the same trend monotonically on its own arch — 26.2, 42.5,
53.7, 89.6 and 163.8 seconds for step lengths of 100, 50, 25, 12.5 and 6.25, with only the
smallest needing no arc adjustment — so halving the step roughly doubles the cost.

What this sweep adds is the other end of the curve. Past a point a larger step stops
paying, because the solver spends its time cutting back and retrying: 34 reductions at a
step of 640 against none at 80. **The cost is U-shaped, not monotonic**, and the cheapest
setting is the largest step that is not provoking many retries.

## 2 and 3. A four-variable design space, solved with Q8

Four designs, each traced with the arc-length solver on twenty eight-node elements,
prescribed arc length 90.7 and total 1684:

| design | radii | steps | time (s) | limit load | u at limit | u reached |
|---|---|---|---|---|---|---|
| uniform | 100, 100, 100, 100 | 19 | 8.8 | 280.3 | −106.2 | −161.7 |
| raised | 88, 112, 112, 88 | 19 | 8.5 | 209.6 | −96.7 | −165.8 |
| flattened | 112, 88, 88, 112 | 19 | 7.0 | 362.4 | −112.0 | −149.5 |
| skewed | 95, 118, 96, 108 | 19 | 8.6 | 193.5 | −95.3 | −162.0 |

All four are clean snap-through within this arc length: the load rises to a limit point,
turns over, and falls while the crown keeps descending. None turns back in displacement.
That matches how the paper describes this structure throughout — "the snap-through behavior
presented in this research".

**Two things the notebook is careful about here.**

*Snap-back is a question of how far you trace.* Every one of these arches reverses
eventually, deep in the post-buckling tail. At the 215° span the uniform arch reverses only
50 mm past its limit point (at u = −165.3, a 28.4 mm reversal); at 270° it takes some 250 mm,
by which point the arch has inverted through itself. Neither the section (R/h from 10 to 29
all reverse) nor the pin position (centroid instead of outer fibre) changes that. So the
prescribed total arc length decides what the figure shows, and the notebook *reports*
whether each design turned back rather than asserting it.

*One design in this space cannot be traced at all.* The obvious skewed choice, 92 at the
first station, runs into a very sharp limit point at an accumulated arc length of 1088 —
the critical eigenvalue drops from 1.7e-3 to 1.2e-4 in one step while the next eigenvalue
stays at 2.7e-1. It is a limit point, not a bifurcation: the critical mode is parallel to
`K⁻¹F`. But no arc-step cutting gets past it — `MaxIter=15`, `MaxCuts=25` and half the arc
step all fail at the same place — and it is a region rather than a knife edge, since 91 and
93 fail identically while 95 traces fine. The notebook uses 95. The paper reports the same
phenomenon, that parts of this design space produce load paths its solver could not follow.

## 4. The objective function

Given a target load path, the inverse problem is to find the radii that reproduce it. The
two curves are not sampled at the same places, so the paper treats both as parametric
curves in **accumulated arc length** and interpolates linearly — either the target onto the
design's arc lengths (**objective function A**) or the design onto the target's
(**objective function B**) — then takes the mean squared error over the load factor and the
crown displacement, its equation (9). Both are scaled by the target curve's range so load
and displacement weigh equally.

> Equation (9) as printed has a **plus** in the displacement term, `(uT + ud)²`. That has to
> be a typesetting slip: with a plus the objective is not zero when the curves coincide, and
> equation (10) of the same paper, its gradient, carries the minus.

The gradient falls out cheaply because **`ds/dx = 0`**. The solver enforces
`‖ΔU‖² + ψ²Δλ² = ℓ²` exactly, with `ℓ` prescribed and only ever divided by `√2` on a retry,
so the arc length reached at each step is a constant of the design and the interpolation
weights are constants in the gradient. That holds right up until the *cut pattern* changes,
where the sample points jump — which is the discontinuity the paper is about.

The target is the fully symmetrical arch, so the global optimum is known. Both objectives
return **f = 0 and |g| = 0 exactly** there. At `[100, 105, 98, 102]`, A gives 0.00254985 and
B gives 0.00262293.

Central differences on one variable, against the analytical gradient:

| step | steps −/+ | central FD, A | rel err | central FD, B | rel err |
|---|---|---|---|---|---|
| 1e-1 | 20/20 | 0.0007037624 | 3.4e-06 | 0.0006989728 | 3.9e-06 |
| 1e-2 | 20/20 | 0.0007037649 | 1.0e-07 | 0.0006989756 | 9.3e-08 |
| 1e-3 | 20/20 | 0.0007037649 | 1.4e-07 | 0.0006989756 | 1.3e-07 |
| 1e-4 | 20/20 | 0.0007037649 | 1.6e-07 | 0.0006989756 | 1.6e-07 |
| 1e-5 | 20/20 | 0.0007037648 | 5.0e-08 | 0.0006989755 | 5.3e-08 |
| 1e-6 | 20/20 | 0.0007037631 | 2.4e-06 | 0.0006989738 | 2.5e-06 |

Analytical: A 0.0007037648, B 0.0006989755. Agreement of `5e-8` at best, comfortably past
the paper's report of six significant figures.

Every perturbation here took the same number of arc steps, so none of these differences
crossed a discontinuity. That is not always so, and the table reports the step counts for
exactly that reason: at the 215° span the same check straddled a cut-pattern change at the
two largest steps and the finite difference came out wrong by one to two orders of
magnitude, in both magnitude and sign — "incorrect in both magnitude and direction", as the
paper puts it. The analytical gradient is unaffected, being the derivative of the branch
actually taken.

## Caveat on the limit load

The peak load factor is **not fully mesh converged**. For the uniform design with 5β
elements:

| elements | free dofs | λ max, fixed arc | λ max, arc scaled by √ndof |
|---|---|---|---|
| 20 | 78 | 237.2 | 237.2 |
| 40 | 158 | 184.5 | 184.7 |
| 60 | 238 | 177.3 | 180.8 |

It is settling — down 22% from 20 to 40 elements, then 4% from 40 to 60 — but 20 elements,
which the sections above use, overestimates by about a third.

This is ordinary discretisation error, not a solver artefact. Scaling the arc length with
`sqrt(ndof)` on refinement changes the answer by under 2%, so the constraint's dependence on
the degree-of-freedom count is *not* responsible — an earlier draft of this file blamed it,
and that was wrong. Refining the arc length on a fixed mesh converges cleanly, and sharing
the crown load across the section rather than at a point moves it by under 0.1%. What is
left is the element: 5β is low order and a deep arch rotating this far needs many of them.

Nothing above depends on it — the sweep holds the mesh fixed and varies only the step, the
design comparison holds the mesh fixed and varies only the shape, and the objective compares
two curves computed on the same mesh with the same settings — but the limit loads should be
read relative to one another, not as converged values.

## Not covered yet

The optimiser half of the paper: the discontinuity study (its Figures 9 and 10, sampling the
objective across radii of 80 to 120 mm), and SLSQP against the two gradient-only methods,
GOSSA and a Modified Subgradient method.

**Where GOSSA lives.** `pmo.py`, the module accompanying Snyman and Wilke, *Practical
Mathematical Optimization* (2nd ed., Springer 2018). It is a single file rather than a
package and is not on PyPI; its original home at `extras.springer.com` now returns 404,
though the Internet Archive holds a copy. It defines `gossa`, `gosda` and `bfgsgo` as
`scipy.optimize.minimize(method=...)` plug-ins, so the notebook's `Objective` drops straight
in. It is Springer copyright and deliberately **not vendored here**. The Modified Subgradient
method needs nothing external — fixed-step steepest descent, no line search, no memory of the
best point seen.

## The cost of a gradient

Table 3 of the paper reports the analytical sensitivity costing about 48 seconds whether
there are 1, 2, 4, 8 or 16 design variables, against forward differences growing 54 → 459.
The point is that the analytical cost does not scale with the number of design variables.
This implementation now shares that property:

| design variables | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 5β arc-length solve with sensitivities (s) | 12.4 | 13.3 | 12.9 | 13.3 |

It did not until recently, and profiling it was worth the trouble. Of one sensitivity step,
5β at 20 elements:

| | 5β | Q8 |
|---|---|---|
| tangent assembly | 42.5 ms | 168.1 ms |
| factorisation | 0.1 ms | 0.3 ms |
| one back substitution | 0.0 ms | 0.0 ms |
| **`dRdx` assembly** | **3432.6 ms** | **15438.0 ms** |

`dRdx` was 99% of it. Two things were wrong, and both are fixed:

**The chain rule was associated the expensive way.** `_dRdXVariable` integrated each
element once per moved nodal co-ordinate *per design variable*, and since the radius spline
moves every node, that was the whole cost — 534 element integrations of which only 152 were
distinct. But in

```
dR/dx_j = sum_a (dR/dX_a) (dX_a/dx_j)
```

the element integral `dR/dX_a` depends on the element and its local degree of freedom
alone; the design variable enters only through the scalar weight. Integrating once per
(element, local dof) and contracting against every variable afterwards is the same
derivative, computed in the other order, at a cost independent of the variable count.
Verified bit-identical against the previous implementation: `dUdx` and `dLdx` agree to
`0.000e+00` for both element types.

**The Jacobian was rebuilt and inverted about seventy times per Gauss point.** `cProfile`
over 534 integrations counted 296,904 calls to `InvJ` and 307,584 to `np.linalg.inv` — 63%
and 44% of the total. It is now kept per quadrature point on the element object.

Together: a 5β sensitivity solve went from 83 s to 16.5 s, Q8 from a comparable factor, and
a *plain* solve from 6.5 s to 3.3 s, since the Jacobian caching helps every assembly. The
whole notebook went from about eight minutes to three.

**Not the adjoint method**, which is the usual answer when cost scales with the number of
design variables. The adjoint replaces one linear solve per variable with a single adjoint
solve — and here the solves were already free, at 0.0 ms. It would also not fit neatly: the
arc-length recursion needs the full `dU/dx` field at each step to carry into the next step's
constraint, so there is nothing to contract to a scalar step by step. A genuine adjoint
would have to be a reverse sweep over the whole traced path, and it still would not avoid
the `dR/dX` integrals, which are where the time goes.

**Still on the table**, if more is wanted: each element is entered once per *local* degree
of freedom, rebuilding `H`, `M` and `G` every time. Integrating all of an element's local
degrees of freedom in one visit is worth roughly another 8× for 5β and 16× for Q8, but it
changes the `Element` contract — `Integrate`/`ResIntegrate` would take a list of degrees of
freedom rather than one.
