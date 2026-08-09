# FEM2D: Finite Element Code for Shape Sensitivities

## Overview
This is a basic finite element code developed during my PhD, focusing on shape sensitivities with respect to load-displacement paths in nonlinear structural problems. The code is designed for small-mesh FEM problems, emphasizing the understanding of finite element methods and the implementation of design sensitivities. The code also includes a Q3 1D element implementation.

## Quick Start
Set up a cantilever, solve it with large deflections, and overlay the deformed mesh on the
undeformed one:

```python
import matplotlib.pyplot as plt

from fem2d.materials import PlaneState
from fem2d.meshers import Mesh
from fem2d.postprocessing import Plotting
from fem2d.solvers import NonLinearSolver

# A cantilever, pinned at the left edge and loaded down at the tip.
Beam = Mesh(E=210000.0, v=0.3, thickness=1.0, plane=PlaneState.Strain, ElementType='5B')
Beam.SimpleBeam(el_num=20, Length=10.0, Height=0.5, Load=-25.0)

# Large deflection solve, load applied over three steps.
NonLinearSolver(Beam, LoadSteps=3, tol=1e-6).Solve()

print(f'tip deflection: {Beam.U[-1, 0]:.4f}')   # -3.1964

# Undeformed mesh with the deformed one over it, fading in the load steps.
fig, ax = plt.subplots(figsize=(9, 2.6))
Plotting.Overlay(Beam, ax=ax, steps=1)
plt.show()
```

Solvers write their results onto the mesh: `Beam.U` is the displacement vector, `Beam.AllU`
every load step, and `Beam.LoadValues` the load factors. Pass `Sensitivity=True` to any solver
and `Beam.dUdx` holds the analytical shape sensitivities as well. Solver progress goes to the
`fem2d.solvers` logger rather than to stdout, so `logging.basicConfig(level=logging.INFO)`
turns it on.

Fields are contoured on the deformed or undeformed mesh:

```python
from fem2d.postprocessing import ContourOptions

Plotting.StressContour(Beam, ContourOptions(Component='vonMises'))
Plotting.StrainContour(Beam, ContourOptions(Component='xy', Deformed=False))
```

Stress and strain are discontinuous between elements, so `Recovery` chooses what happens at
the nodes: `'extrapolate'` (default) samples at the Gauss points where the stress is most
accurate and fits back to the nodes before averaging, `'average'` evaluates at the nodes and
averages there, and `'none'` leaves each element with its own values so the jumps stay
visible. `NodalField` returns the same numbers without plotting.

## Features
- Supports **three element types**, **three solvers**, and **three meshers** for simple structural problems.
- Developed with a focus on **nonlinear structural analysis** and **design sensitivity analysis**.

## Solvers
The code includes the following solvers:
- **Linear**: Solves small deformation elastic problems.
- **Nonlinear**: Handles large deformation problems with material and geometric nonlinearities.
- **Arc-Length**: Implements an arc-length continuation method for tracing equilibrium paths beyond limit points.

## Element Types
- **Q4**: Four-node quadrilateral element.
- **Q8**: Eight-node quadrilateral element.
- **5β**: Five-parameter assumed-stress element.

## Mesh Generators
- **Cantilever**: Generates a cantilever beam mesh.
- **Deep Semi-Circular Arch**: Creates a deep arch mesh.
- **Curved Beam**: Mesh for curved beam structures.

## Example Usage
The `notebooks/` directory contains worked examples, each executed with its output stored:

- **`Example_solvers.ipynb`** — every element and solver combination on a cantilever, deformed
  shapes, load paths, and the four meshers.
- **`Example_sensitivity.ipynb`** — shape sensitivities for every element and solver, each checked
  against a central finite difference, plus the load factor sensitivity along an arc length path.
- **`Example_optimisation.ipynb`** — shape optimisation with `scipy.optimize.minimize`, finding the
  cantilever length that deflects to a target under a fixed load, driven by the analytical gradient.
- **`Example_custom_element.ipynb`** — how to add your own element. Builds a four node quadrilateral
  interpolated with sin² and cos², which turns out to be Q4 in disguise but with an integrand Gauss
  quadrature cannot integrate exactly, so it doubles as a lesson in choosing a quadrature rule.




