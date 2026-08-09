import numpy as np

from fem2d.elements.quad4 import Q4


class FiveBeta(Q4):
    '''
    Five parameter assumed stress quadrilateral (Pian-Sumihara).

    A Q4 whose stress field is interpolated independently of the
    displacements, which is what removes the shear locking Q4 suffers from.
    '''

    #: Number of assumed stress parameters, the beta vector.
    NumBeta = 5


    def P(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        P : Interpolation Matrix for the assumed Stress Element.

        This is the classical Pian-Sumihara stress field, in which the
        coefficients of the xi derivative (a1, b1) multiply eta and those of
        the eta derivative (a3, b3) multiply xi. The pairing is tied to the
        convention used by dN, so the two must only ever be changed together;
        crossing them makes the element rank deficient.
        '''
        mat = 1/4*np.array([[-1, 1, 1, -1],
                             [1, -1, 1, -1],
                             [-1, -1, 1, 1]])

        A = mat @ self.NodeCoor[:,[0]]
        B = mat @ self.NodeCoor[:,[1]]

        a1, a3 = A[0,0], A[2, 0]
        b1, b3 = B[0, 0], B[2, 0]

        if self.LinearFlag:

            P = np.array([[1, 0, 0, a1**2*eta, a3**2*xi],
                          [0, 1, 0, b1**2*eta, b3**2*xi],
                          [0, 0, 1, a1*b1*eta, a3*b3*xi]])

        else:

             P = np.array([[1, 0, 0, a1**2*eta, a3**2*xi],
                          [0, 1, 0, b1**2*eta, b3**2*xi],
                          [0, 0, 1, a1*b1*eta, a3*b3*xi],
                          [0, 0, 1, a1*b1*eta, a3*b3*xi]])

        return P

    def Ge(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        G : Variable needed for stiffness matrix
           or Residual and Tangenet.

        '''

        if self.LinearFlag:

            G = self.B(xi, eta).T @ self.P(xi, eta) * self.detJ(xi, eta)

        else:

            G = self.B(xi, eta).T @ self.Fmat(xi, eta) @ self.P(xi, eta) * self.detJ(xi, eta)

        return G

    def He(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        H : Variable needed for stiffness matrix.

        '''

        H = self.P(xi, eta).T @ np.linalg.inv(self.C()) @ self.P(xi, eta) * self.detJ(xi,eta)

        return H

    def Me(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        M : Variable needed for Residual and Tangent.

        '''

        M = self.P(xi, eta).T @ self.Evec(xi, eta) * self.detJ(xi, eta)

        return M

    def Le(self, xi, eta, svec):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.
        svec : Assumed stress vector P @ Beta at the local co-ordinates.

        Returns
        -------
        L : Variable needed for Residual and Tangent.

        '''
        L = self.B(xi, eta).T @ self.StressMat(svec) @ self.B(xi, eta) * self.detJ(xi, eta)

        return L

    def StiffMatrix(self, GaussPoints = None):
        '''
        Parameters
        ----------
        GaussPoints : Select the number of Gauss points.
                      The default is 2.

        Returns
        -------
        StiffMatrix : (8x8) Element Stiffness Matrix.
        '''

        G = np.zeros((self.NumDOF, self.NumBeta))
        H = np.zeros((self.NumBeta, self.NumBeta))

        gp, gw = self.GaussPointsAndWeights(GaussPoints) #gauss points and weights

        for Xi, Wxi in zip(gp, gw, strict=True):

            for Eta, Weta in zip(gp, gw, strict=True):

                G += self.Ge(Xi, Eta) * Wxi * Weta

                H += self.He(Xi, Eta) * Wxi * Weta

        StiffMatrix = self.t * G @ np.linalg.inv(H) @ G.T

        return StiffMatrix

    def ResTangent(self, GaussPoints = None):
        '''
        Parameters
        ----------
        GaussPoints : Number of Gauss Points
            DESCRIPTION. The default is 2.

        Returns
        -------
        TangentMatrix : Integrated Tangent Stiffness Matrix.
        ResidualVector : Integrated residual Vector.

        '''
        G = np.zeros((self.NumDOF, self.NumBeta))
        H = np.zeros((self.NumBeta, self.NumBeta))
        M = np.zeros((self.NumBeta, 1))
        L = np.zeros((self.NumDOF, self.NumDOF))

        gp, gw = self.GaussPointsAndWeights(GaussPoints) #gauss points and weights

        for Xi, Wxi in zip(gp, gw, strict=True):

            for Eta, Weta in zip(gp, gw, strict=True):

                M += self.Me(Xi, Eta) * Wxi * Weta

                H += self.He(Xi, Eta) * Wxi * Weta

                G += self.Ge(Xi, Eta) * Wxi * Weta

        B = np.linalg.inv(H) @ M

        # The geometric term is the derivative of G(u) @ Beta, so it must be
        # evaluated with the assumed stress P @ Beta. Using the displacement
        # derived stress C @ Evec instead leaves the tangent inconsistent,
        # which costs quadratic Newton convergence and corrupts every
        # sensitivity that solves with this matrix.
        for Xi, Wxi in zip(gp, gw, strict=True):

            for Eta, Weta in zip(gp, gw, strict=True):

                L += self.Le(Xi, Eta, self.P(Xi, Eta) @ B) * Wxi * Weta

        ResidualVector = self.t * G @ B

        TangentMatrix = self.t * (L + G @ np.linalg.inv(H) @ G.T)

        return TangentMatrix, ResidualVector
