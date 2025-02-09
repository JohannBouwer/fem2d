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
The repository includes example files demonstrating the usage of different solvers, element types, and meshers. These examples help users understand:
- How to define input parameters.
- How to select different solvers and element types.
- How to visualize and interpret results.

For any questions or contributions, feel free to reach out!



