# FEM2D: Finite Element Code for Shape Sensitivities

## Overview
This is a basic finite element code developed during my PhD, focusing on shape sensitivities with respect to load-displacement paths in nonlinear structural problems. The code is designed for small-mesh FEM problems, emphasizing the understanding of finite element methods and the implementation of design sensitivities. The code also includes a Q3 1D element implementation.

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




