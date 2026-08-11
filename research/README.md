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
