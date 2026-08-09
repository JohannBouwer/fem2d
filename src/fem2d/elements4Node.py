"""Compatibility shim. Elements now live in fem2d.elements."""

from fem2d.elements.assumed_stress import FiveBeta
from fem2d.elements.quad4 import Q4

__all__ = ['Q4', 'FiveBeta']
