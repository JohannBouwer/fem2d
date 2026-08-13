# Research

Short numerical walk-throughs of the papers this code was written for. One notebook per
paper, each committed with its output so it reads without being run.

## `ArcLengthControlledShapeDesign.ipynb`

> Bouwer, J. M., Kok, S. and Wilke, D. N. (2023). *Challenges and solutions to arc-length
> controlled structural shape design problems.* Mechanics Based Design of Structures and
> Machines, **51**(7), 4088–4119.
> [doi:10.1080/15397734.2021.1950549](https://doi.org/10.1080/15397734.2021.1950549)

A 270° deep circular arch whose shape is four radii, loaded at the crown and traced with the
arc-length solver. The first half is analysis: why large arc-length steps pay for themselves,
four designs and their load-deflection paths on Q8 elements. The second half is the design
problem — the paper's curve-matching objective, equations (9) and (10), with its analytical
gradient checked against central differences to eight figures. The point is the last two
sections: the solver's adaptive arc stepping makes the objective discontinuous, so SLSQP
declares success 10.7 mm from a known optimum, while ADAM, which never reads the objective
value, halves that error on the identical problem. Still to come are the paper's own
discontinuity study and its GOSSA and Modified Subgradient optimisers.

Runs in about two hours, nearly all of it the two optimisers.

## `SurrogateBasedShapeDesign.ipynb`

> Bouwer, J., Wilke, D. N. and Kok, S. (2024). *A novel and fully automated coordinate system
> transformation scheme for near optimal surrogate construction.* Computer Methods in Applied
> Mechanics and Engineering, **419**, 116648.
> [doi:10.1016/j.cma.2023.116648](https://doi.org/10.1016/j.cma.2023.116648)

The load path itself as a surrogate, rather than an objective computed from it. Needs the
optional `ge_rbf` dependency (`uv sync --extra surrogate`); the glue between the solver's output
and its trajectory API is [`fem2d.surrogate`](../src/fem2d/surrogate.py).

The crown load factor and the crown displacement are each fitted as a function of the design
radii **and the accumulated arc length**, so one traced path contributes a sample per arc step
rather than a single sample. That is the standard way round — an objective becomes a calculation
on top of the fitted responses, and can be changed or replaced without sampling anything again.
Sections 1–5 use one design variable and four solves, few enough that both surfaces can be drawn
in 3D and checked against a fifth design held back. Section 6 repeats it with three variables and
fifteen curves, where nothing can be drawn and accuracy is shown by laying predicted load paths
over solved ones. Fitting the responses over arc length is what needs `ArcLengthSolver`'s
`dLds` and `dUds_All`; the notebook checks them against the two identities that make them
self-checking before using them.

Optimising over the fitted responses is the next step and is deliberately not here.

Runs in about 15 minutes, nearly all of it the 23 arc-length solves carrying sensitivities.

