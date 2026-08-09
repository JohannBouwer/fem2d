"""Shape derivatives of the element quantities.

ShapeDerivative differentiates everything Element computes with respect to the
nodal co-ordinates. It is written against the Element interface alone, so it
applies unchanged to any element: pair it with a concrete element and the
derivative class is complete.

    class dMyElementdX(ShapeDerivative, MyElement):
        pass
"""

import numpy as np

from fem2d.elements.assumed_stress import FiveBeta
from fem2d.elements.base import Element
from fem2d.elements.quad4 import Q4
from fem2d.elements.quad8 import Q8


class ShapeDerivative(Element):
    '''
    Mixin adding the nodal co-ordinate derivatives to any Element.

    Mix in front of the element being differentiated so that QuadOrder,
    NumNodes and the shape functions are inherited from it. That inheritance
    is what stops a derivative being integrated at a different quadrature
    order than the quantity it differentiates.
    '''


    def dJdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.
        DOF : Local degree of freedom to take the derivative w.r.t .

        Returns
        -------
        Derivative of the Jacobian w.r.t the nodal Coordinates.

        '''

        x = np.zeros((self.NumNodes, 2))
        x[int(DOF//2), DOF%2] = 1

        djdx = self.ShapeDerivatives(xi, eta) @ x

        return djdx

    def dInvJdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        Derivative of the Inverse of the Jacobian w.r.t
        the nodal Coordinates.
        '''

        # Use the known identity dM^-1dx = -M^-1 dMdX M^-1

        dinvjdx = np.linalg.multi_dot([-1*self.InvJ(xi, eta), self.dJdX(xi, eta, DOF), self.InvJ(xi, eta)])


        return dinvjdx

    def ddetJdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        Derivative of the determinant of the Jacobian w.r.t the
        the nodal coordinates.

        This uses the Jacobi's formula':
           https://en.wikipedia.org/wiki/Jacobi%27s_formula
        '''

        ddetjdx = self.detJ(xi, eta)*np.trace(self.InvJ(xi, eta) @ self.dJdX(xi, eta, DOF))

        return ddetjdx

    def dBdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        Derivative of the strain matrix w.r.t
        the nodal co-ordinates.

        '''

        dinvjdx = self.dInvJdX(xi, eta, DOF)

        if self.LinearFlag:

            dMapJacobian = np.array([[dinvjdx[0,0], dinvjdx[0,1], 0, 0],
                                     [0, 0, dinvjdx[1,0], dinvjdx[1,1]],
                                     [dinvjdx[1,0], dinvjdx[1,1], dinvjdx[0,0], dinvjdx[0,1]]])

        else:

            dMapJacobian = np.array([[dinvjdx[0,0], dinvjdx[0,1], 0, 0],
                                     [0, 0, dinvjdx[1,0], dinvjdx[1,1]],
                                     [dinvjdx[1,0], dinvjdx[1,1], 0, 0],
                                     [0, 0, dinvjdx[0,0], dinvjdx[0,1]]])

        db = dMapJacobian.dot(self.dN(xi, eta))

        return db

    def dFvecdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.
        DOF : Local degree of freedom to take
              the derivative w.r.t .

        Returns
        -------
        dfvecdx : Derivative of the derformation gradient.

        '''
        dfvecdx = self.dBdX(xi, eta, DOF).dot(self.U)

        return dfvecdx

    def dFmatdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.
        DOF : Local degree of freedom to take
              the derivative w.r.t .

        Returns
        -------
        dfmatdx : Derivative of the derformation gradient in matrix form.

        '''

        dfvecdx = self.dFvecdX(xi, eta, DOF)

        dfmatdx = np.array([[dfvecdx[0,0], 0, 0.5*dfvecdx[2,0], 0.5*dfvecdx[2,0]],
                            [0, dfvecdx[1,0], 0.5*dfvecdx[3,0], 0.5*dfvecdx[3,0]],
                            [0, dfvecdx[2,0], 0.5*dfvecdx[0,0], 0.5*dfvecdx[0,0]],
                            [dfvecdx[3,0], 0, 0.5*dfvecdx[1,0], 0.5*dfvecdx[1,0]]])

        return dfmatdx

    def dEvecdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.
        DOF : Local degree of freedom to take
              the derivative w.r.t .

        Returns
        -------
        devecdx : Derivative of the Green-Lagrane strain.

        '''

        devecdx = 0.5*(self.dFmatdX(xi, eta, DOF).T.dot(self.Fvec(xi, eta)) +
                    self.Fmat(xi, eta).T.dot(self.dFvecdX(xi, eta, DOF)))

        return devecdx

    def dSvecdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.
        DOF : Local degree of freedom to take
              the derivative w.r.t .

        Returns
        -------
        dsvecdx : derivative of the second Piola-Kirchhoff stress vector.
        '''

        dsvecdx = self.C().dot(self.dEvecdX(xi, eta, DOF))

        return dsvecdx

    def dKdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.
        DOF : Local degree of freedom to take
              the derivative w.r.t .

        Returns
        -------
        Ke : Derivative of Element Stiffness Matrix as a function of the local co-ordinates.

        K = BT * C * B * detJ
        dK = dBT * C * B * detJ
           + BT * C * dB * detJ
           + BT * C * B * ddetJ
        '''

        dKedX = (np.linalg.multi_dot([self.dBdX(xi, eta, DOF).T, self.C(), self.B(xi, eta)])*self.detJ(xi, eta)
              +  np.linalg.multi_dot([self.B(xi, eta).T, self.C(), self.dBdX(xi, eta, DOF)])*self.detJ(xi, eta)
              +  np.linalg.multi_dot([self.B(xi, eta).T, self.C(), self.B(xi, eta)])*self.ddetJdX(xi, eta, DOF))

        return dKedX

    def dRdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.
        DOF : Local degree of freedom to take
              the derivative w.r.t .

        Returns
        -------
        drdx : Derivative of the Element Residual as a function of the local coordinates.

        '''
        detJ = self.detJ(xi, eta)

        drdx = (np.linalg.multi_dot([self.dBdX(xi, eta, DOF).T, self.Fmat(xi, eta), self.Svec(xi, eta)])*detJ
             +  np.linalg.multi_dot([self.B(xi, eta).T, self.dFmatdX(xi, eta, DOF), self.Svec(xi, eta)])*detJ
             +  np.linalg.multi_dot([self.B(xi, eta).T, self.Fmat(xi, eta), self.dSvecdX(xi, eta, DOF)])*detJ
             +  np.linalg.multi_dot([self.B(xi, eta).T, self.Fmat(xi, eta), self.Svec(xi, eta)])
                *self.ddetJdX(xi, eta, DOF))

        return drdx

    def Integrate(self, DOF, GaussPoints = None):
        '''
        Parameters
        ----------
        DOF : Local degree of freedom to take
              the derivative w.r.t .
        GaussPoints : TYPE, int
            Number of Gauss Points to use. The default is 2.

        Returns
        -------
        SensMatrix : Derivivite of the element stiffness matrix.

        '''
        SensMatrix = np.zeros((self.NumDOF, self.NumDOF))
        gp, gw = self.GaussPointsAndWeights(GaussPoints) #gauss points and weights

        for Xi, Wxi in zip(gp, gw, strict=True):

            for Eta, Weta in zip(gp, gw, strict=True):

                SensMatrix += self.t * self.dKdX(Xi, Eta, DOF) * Wxi * Weta

        return SensMatrix

    def ResIntegrate(self, DOF, GaussPoints = None):
        '''
        Parameters
        ----------
        DOF : Local degree of freedom to take
              the derivative w.r.t.
        GaussPoints : TYPE, int
            Number of Gauss Points to use. The default is 2.

        Returns
        -------
        SensResidual :  Derivivite of the element residual vector..

        '''
        SensResidual = np.zeros((self.NumDOF, 1))
        gp, gw = self.GaussPointsAndWeights(GaussPoints) #gauss points and weights

        for Xi, Wxi in zip(gp, gw, strict=True):

            for Eta, Weta in zip(gp, gw, strict=True):

                SensResidual += self.t * self.dRdX(Xi, Eta, DOF) * Wxi * Weta

        return SensResidual


class dQ4dX(ShapeDerivative, Q4):
    '''Shape derivatives of the four node quadrilateral.'''


class dQ8dX(ShapeDerivative, Q8):
    '''Shape derivatives of the eight node quadrilateral.'''


class d5BdX(ShapeDerivative, FiveBeta):

    def dPdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.
        DOF : Local degree of freedom to take
              the derivative w.r.t.

        Returns
        -------
        dPdX : derivative of the Interpolation Matrix for the assumed Stress Element.
        '''

        mat = 1/4*np.array([[-1, 1, 1, -1],
                             [1, -1, 1, -1],
                             [-1, -1, 1, 1]])

        A = mat @ self.NodeCoor[:,[0]]
        B = mat @ self.NodeCoor[:,[1]]

        a1, a3 = A[0,0], A[2, 0]
        b1, b3 = B[0, 0], B[2, 0]

        if DOF%2 == 0:

            dX = np.zeros((self.NumNodes, 1))
            dX[DOF//2, 0] = 1
            dA = mat @ dX

            da1, da3 = dA[0,0], dA[2, 0]

            #xi/eta pairing follows FiveBeta.P
            dP = np.array([[0, 0, 0, 2*a1*da1*eta, 2*a3*da3*xi],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, b1*da1*eta, b3*da3*xi]])


        else:

            dX = np.zeros((self.NumNodes, 1))
            dX[DOF//2, 0] = 1
            dB = mat @ dX

            db1, db3 = dB[0,0], dB[2, 0]

            #xi/eta pairing follows FiveBeta.P
            dP = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 2*b1*db1*eta, 2*b3*db3*xi],
                           [0, 0, 0, a1*db1*eta, a3*db3*xi]])


        if self.LinearFlag:

            dPdX = dP

        else:

            dPdX = np.zeros((4, self.NumBeta))

            dPdX[:-1,:] = dP
            dPdX[-1,:] = dP[-1,:]

        return dPdX

    def dHedX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.
        DOF : Local degree of freedom to take
              the derivative w.r.t.

        Returns
        -------
        dHdX : derivative of the local variable He for the assumed stress element.
        '''

        dHdX = (self.P(xi, eta).T @ np.linalg.inv(self.C()) @ self.P(xi, eta) * self.ddetJdX(xi, eta, DOF)
                + self.dPdX(xi, eta, DOF).T @ np.linalg.inv(self.C()) @ self.P(xi, eta) * self.detJ(xi, eta)
                + self.P(xi, eta).T @ np.linalg.inv(self.C()) @ self.dPdX(xi, eta, DOF) * self.detJ(xi, eta))

        return dHdX

    def dMedX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.
        DOF : Local degree of freedom to take
              the derivative w.r.t.

        Returns
        -------
        dMdX : derivative of the local variable Me for the assumed stress element.
        '''
        #0.5*(Fvec_to_Fmat(d_Fec)'*Fvec + Fmat'*d_Fec)
        #t1 = 0.5*self.dFmatdX(xi, eta, DOF).T @ self.Fvec(xi, eta) + self.Fmat(xi, eta).T @ self.dFvecdX(xi, eta, DOF)

        dMdX = (self.P(xi, eta).T @ self.dEvecdX(xi, eta, DOF) * self.detJ(xi, eta)
                + self.P(xi, eta).T @ self.Evec(xi, eta) * self.ddetJdX(xi, eta, DOF)
                + self.dPdX(xi, eta, DOF).T @ self.Evec(xi, eta) * self.detJ(xi, eta))

        return dMdX

    def dGedX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.
        DOF : Local degree of freedom to take
              the derivative w.r.t.

        Returns
        -------
        dGdX : derivative of the local variable Ge for the assumed stress element.
        '''

        if self.LinearFlag:

            dGdX = (self.dBdX(xi, eta, DOF).T @ self.P(xi, eta) * self.detJ(xi, eta)
                    + self.B(xi, eta).T @ self.dPdX(xi, eta, DOF) * self.detJ(xi, eta)
                    + self.B(xi, eta).T @ self.P(xi, eta) * self.ddetJdX(xi, eta, DOF))
        else:

            dGdX = (self.dBdX(xi, eta, DOF).T @ self.Fmat(xi, eta) @ self.P(xi, eta) * self.detJ(xi, eta)
                    + self.B(xi, eta).T @ self.dFmatdX(xi, eta, DOF) @ self.P(xi, eta) * self.detJ(xi, eta)
                    + self.B(xi, eta).T @ self.Fmat(xi, eta) @ self.dPdX(xi, eta, DOF) * self.detJ(xi, eta)
                    + self.B(xi, eta).T @ self.Fmat(xi, eta) @ self.P(xi, eta) * self.ddetJdX(xi, eta, DOF))

        return dGdX

    def Integrate(self, DOF, GaussPoints = None):
        '''
        Parameters
        ----------
        DOF : Local degree of freedom to take
              the derivative w.r.t .
        GaussPoints : TYPE, int
            Number of Gauss Points to use. The default is 2.

        Returns
        -------
        SensMatrix : Derivivite of the element stiffness matrix.

        '''

        G = np.zeros((self.NumDOF, self.NumBeta))
        H = np.zeros((self.NumBeta, self.NumBeta))

        dG = np.zeros((self.NumDOF, self.NumBeta))
        dH = np.zeros((self.NumBeta, self.NumBeta))

        gp, gw = self.GaussPointsAndWeights(GaussPoints) #gauss points and weights

        for Xi, Wxi in zip(gp, gw, strict=True):

            for Eta, Weta in zip(gp, gw, strict=True):

                G += self.Ge(Xi, Eta) * Wxi * Weta

                H += self.He(Xi, Eta) * Wxi * Weta

                dH += self.dHedX(Xi, Eta, DOF) * Wxi * Weta

                dG += self.dGedX(Xi, Eta, DOF) * Wxi * Weta


        SensStiffMatrix = self.t * (dG @ np.linalg.inv(H) @ G.T
                                    + G @ (-1*np.linalg.inv(H) @ dH @ np.linalg.inv(H)) @ G.T
                                    + G @ np.linalg.inv(H) @ dG.T)

        return SensStiffMatrix

    def ResIntegrate(self, DOF, GaussPoints = None):
        '''
        Parameters
        ----------
        DOF : Local degree of freedom to take
              the derivative w.r.t.
        GaussPoints : TYPE, int
            Number of Gauss Points to use. The default is 2.

        Returns
        -------
        SensResidual :  Derivivite of the element residual vector..

        '''
        SensResidual = np.zeros((self.NumDOF, 1))
        gp, gw = self.GaussPointsAndWeights(GaussPoints) #gauss points and weights

        dG = np.zeros((self.NumDOF, self.NumBeta))
        dH = np.zeros((self.NumBeta, self.NumBeta))
        dM = np.zeros((self.NumBeta, 1))

        G = np.zeros((self.NumDOF, self.NumBeta))
        H = np.zeros((self.NumBeta, self.NumBeta))
        M = np.zeros((self.NumBeta, 1))

        for Xi, Wxi in zip(gp, gw, strict=True):

            for Eta, Weta in zip(gp, gw, strict=True):

                dM += self.dMedX(Xi, Eta, DOF) * Wxi * Weta

                dH += self.dHedX(Xi, Eta, DOF) * Wxi * Weta

                H += self.He(Xi, Eta) * Wxi * Weta

                M += self.Me(Xi, Eta) * Wxi * Weta

        B = np.linalg.inv(H) @ M
        dB = (-1*np.linalg.inv(H) @ dH @ np.linalg.inv(H)) @ M + np.linalg.inv(H) @ dM

        for Xi, Wxi in zip(gp, gw, strict=True):

            for Eta, Weta in zip(gp, gw, strict=True):

                G += self.Ge(Xi, Eta) * Wxi * Weta

                dG += self.dGedX(Xi, Eta, DOF) * Wxi * Weta

        SensResidual = self.t*dG @ B + self.t*G @ dB

        return SensResidual
