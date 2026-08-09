import numpy as np

from fem2d.elements.base import Element


class Q8(Element):
    '''
    Eight node serendipity quadrilateral.

    Local node order is the four corners counter clockwise from (-1, -1),
    followed by the four mid side nodes starting on the (eta = -1) edge.
    '''

    NumNodes = 8

    QuadOrder = 3

    def ShapeFunctions(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        (8,) array of shape function values.
        '''

        return np.array([-0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
                         -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
                         -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
                         -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
                         0.5 * (1 - xi**2) * (1 - eta),
                         0.5 * (1 + xi) * (1 - eta**2),
                         0.5 * (1 - xi**2) * (1 + eta),
                         0.5 * (1 - xi) * (1 - eta**2)])

    def ShapeDerivatives(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        (2, 8) array of shape function derivitives, row 0 with respect to xi
        and row 1 with respect to eta.
        '''

        return np.array([[-0.25 * (eta - 1)*(eta + 2*xi),
                          -0.25 * (-eta + 2*xi)*(eta - 1),
                          -0.25 * (-eta - 2*xi)*(eta + 1),
                          -0.25 * (eta + 1)*(eta - 2*xi),
                          0.5 * 2*xi*(eta - 1),
                          0.5 * (1 - eta**2),
                          0.5 * -2*xi*(eta + 1),
                          0.5 * (eta**2 - 1)],
                         [-0.25 * (2*eta + xi)*(xi - 1),
                          -0.25 * (-2*eta + xi)*(xi + 1),
                          -0.25 * (-2*eta - xi)*(xi + 1),
                          -0.25 * (2*eta - xi)*(xi - 1),
                          0.5 * (xi**2 - 1),
                          0.5 * -2*eta*(xi + 1),
                          0.5 * (1 - xi**2),
                          0.5 * 2*eta*(xi - 1)]])
