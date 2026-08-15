# FEM2D: Finite Element Code for Shape Sensitivities

## Overview
This is a basic finite element code developed during my PhD, focusing on shape sensitivities with respect to load-displacement paths in nonlinear structural problems. The code is designed for small-mesh FEM problems, emphasizing the understanding of finite element methods and the implementation of design sensitivities. The code also includes a Q3 1D element implementation.

## Installation
Needs Python 3.12 or newer.

### With uv
```bash
git clone https://github.com/JohannBouwer/fem2D.git
cd fem2D
uv sync
```

`uv sync` creates `.venv` and installs the exact versions pinned in `uv.lock`, including
`fem2d` itself in editable mode. Prefix commands with `uv run` and there is no environment to
activate:

```bash
uv run python -c "from fem2d.meshers import Mesh; print('ok')"
```

### With pip
```bash
git clone https://github.com/JohannBouwer/fem2D.git
cd fem2D
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

pip resolves dependencies afresh rather than reading `uv.lock`, so you may get newer versions
than the ones this was tested against. `-e` keeps the install pointing at `src/`, so edits
take effect without reinstalling.

### Running the notebooks
`ipykernel` is a dependency, so `notebooks/` and `research/` open straight away in an editor
that brings its own front end, such as VS Code. JupyterLab itself is deliberately not a
dependency; add it for that run only with

```bash
uv run --with jupyterlab jupyter lab
```

or `pip install jupyterlab` in the pip environment.

### The `surrogate` extra
[`fem2d.surrogate`](src/fem2d/surrogate.py) fits a solved load path as a smooth function of the
shape variables and the arc length together, using the sampled sensitivities in both
directions. `ge_rbf` handles trajectory data natively, so this is only the glue between the
solver's output and its API. It needs
[`ge_rbf`](https://github.com/JohannBouwer/GE_RBF), which is optional because it pulls in
scikit-learn that nothing else here uses, and because the surrogates are one way of spending
the sensitivities rather than part of the finite element code:

```bash
uv sync --extra surrogate
```

`ge_rbf` is not on PyPI, so `pyproject.toml` gives uv its git location. pip does not read that
section and would go looking on PyPI, so in the pip environment install it first and the extra
second, by which point the requirement is already satisfied:

```bash
pip install git+https://github.com/JohannBouwer/GE_RBF.git
pip install -e ".[surrogate]"
```

Nothing else imports it: `import fem2d` works whether or not the extra is installed, and
`import fem2d.surrogate` without it raises an `ImportError` naming the command above.
[`research/SurrogateBasedShapeDesign.ipynb`](research/SurrogateBasedShapeDesign.ipynb) and
[`research/SurrogateBasedOptimisation.ipynb`](research/SurrogateBasedOptimisation.ipynb) are
what use it.

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
every load step, and `Beam.LoadValues` the load factors. `ArcLengthSolver` adds
`Beam.ArcValues`, the arc length accumulated at each stored point, which is what a load path
is parametrised by when two of them are compared. Pass `Sensitivity=True` to any solver
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

## License
MIT — see [LICENSE](LICENSE). Use it, change it, build on it, commercially or otherwise; just
keep the copyright notice. It comes with no warranty.

## Research
The [`research/`](research/) directory reproduces results from the work this code was written
for. See [`research/README.md`](research/README.md) for what each notebook shows.

> Bouwer, J. M., Kok, S. and Wilke, D. N. (2023). *Challenges and solutions to arc-length
> controlled structural shape design problems.* Mechanics Based Design of Structures and
> Machines, **51**(7), 4088–4119.
> [doi:10.1080/15397734.2021.1950549](https://doi.org/10.1080/15397734.2021.1950549)

> Bouwer, J., Wilke, D. N. and Kok, S. (2024). *A novel and fully automated coordinate system
> transformation scheme for near optimal surrogate construction.* Computer Methods in Applied
> Mechanics and Engineering, **419**, 116648.
> [doi:10.1016/j.cma.2023.116648](https://doi.org/10.1016/j.cma.2023.116648)

> Bouwer, J. M. (2023). *The shape optimisation of compliant structures to produce a desired
> snap-through load displacement path.* PhD thesis, University of Pretoria.
> [repository.up.ac.za](https://repository.up.ac.za/items/db54b751-f3d3-4387-914a-27c46a2bcc81)




