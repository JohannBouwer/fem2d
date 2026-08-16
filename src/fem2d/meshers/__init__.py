"""Mesh generators.

A mesh holds the material state, the node and element arrays, the boundary
conditions, the applied load and the shape sensitivities. MeshBase supplies
everything except the geometry, so a new mesh is a node array, a connectivity
array, and calls to the boundary condition and load API. See base.py, or
notebooks/Example_custom_mesh.ipynb for a worked example.
"""

from fem2d.meshers.base import MeshBase, MeshWarning
from fem2d.meshers.eight_node import Q8Mesh
from fem2d.meshers.four_node import Mesh

__all__ = ['MeshBase', 'MeshWarning', 'Mesh', 'Q8Mesh']
