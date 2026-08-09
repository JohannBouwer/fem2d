"""Element library.

Concrete elements, their shape derivative counterparts, and the registry that
lets a mesh name them. See registry.py for how to plug in your own.
"""

from fem2d.elements.assumed_stress import FiveBeta
from fem2d.elements.base import Element
from fem2d.elements.quad4 import Q4
from fem2d.elements.quad8 import Q8
from fem2d.elements.registry import LookupElement, RegisteredElements, RegisterElement
from fem2d.elements.shape_derivative import ShapeDerivative, d5BdX, dQ4dX, dQ8dX

RegisterElement('Q4', Q4, dQ4dX)
RegisterElement('5B', FiveBeta, d5BdX)
RegisterElement('Q8', Q8, dQ8dX)

__all__ = [
    'Element', 'ShapeDerivative',
    'Q4', 'Q8', 'FiveBeta',
    'dQ4dX', 'dQ8dX', 'd5BdX',
    'RegisterElement', 'RegisteredElements', 'LookupElement',
]
