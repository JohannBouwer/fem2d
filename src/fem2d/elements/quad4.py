import numpy as np

from fem2d.elements.base import Element


class Q4(Element):
    '''
    Four node quadrilateral with bilinear shape functions.

    Local node order is counter clockwise from the (-1, -1) corner.
    '''

    NumNodes = 4

    QuadOrder = 2

    LocalNodes = np.array([[-1.0, -1.0],
                           [ 1.0, -1.0],
                           [ 1.0,  1.0],
                           [-1.0,  1.0]])

    def ShapeFunctions(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        (4,) array of shape function values.
        '''

        return np.array([1/4 * (1 - xi)*(1 - eta),
                         1/4 * (1 + xi)*(1 - eta),
                         1/4 * (1 + xi)*(1 + eta),
                         1/4 * (1 - xi)*(1 + eta)])

    def ShapeDerivatives(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        (2, 4) array of shape function derivatives, row 0 with respect to xi
        and row 1 with respect to eta.
        '''

        return np.array([[-1/4 * (1 - eta), 1/4 * (1 - eta), 1/4 * (1 + eta), -1/4 * (1 + eta)],
                         [-1/4 * (1 - xi), -1/4 * (1 + xi), 1/4 * (1 + xi), 1/4 * (1 - xi)]])
