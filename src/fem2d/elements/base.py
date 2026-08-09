"""Abstract base class for every element in the library.

A concrete element supplies its geometry only: how many nodes it has, what
quadrature order it wants, and its shape functions and their local derivatives.
Everything else, from the Jacobian through to the tangent stiffness, is derived
here and shared by every element.

To add an element, subclass Element, set NumNodes and QuadOrder, and implement
ShapeFunctions and ShapeDerivatives. See fem2d/elements/quad4.py for the
smallest complete example.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from fem2d.materials import ConstitutiveMatrix


class Element(ABC):

    #: Number of nodes on the element. Fixes the degree of freedom count.
    NumNodes: ClassVar[int]

    #: Default Gauss points per direction. The matching shape derivative class
    #: inherits it, so a derivative can never be integrated at a different
    #: order than the quantity it differentiates.
    QuadOrder: ClassVar[int]

    def __init__(self, NodeCoor, t, E, v, plane, LinearFlag = True, U = None):
        '''
        Parameters
        ----------
        NodeCoor : array of gloabl nodal co-oridinates.
        t : thickness of the element.
        E : Youngs Modulous.
        v : Poissons ratio.
        plane : even = Plane Stress, odd = Plane Strain.
        LinearFlag; Changes element to nonlinear implementation.
        U: Nodal displacements for element in Nonlinear implementation.
        '''
        self.NodeCoor = NodeCoor
        self.t = t
        self.E = E
        self.v = v
        self.plane = plane
        self.LinearFlag = LinearFlag
        self.U = U

        return

    @property
    def NumDOF(self):
        '''
        Returns
        -------
        Number of degrees of freedom on the element, two per node.
        '''

        return 2*self.NumNodes

    @abstractmethod
    def ShapeFunctions(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        (NumNodes,) array of shape function values, in local node order.
        '''

    @abstractmethod
    def ShapeDerivatives(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        (2, NumNodes) array of shape function derivitives, row 0 with respect
        to xi and row 1 with respect to eta.
        '''

    def GuassPointsAndWeights(self, GuassPoints = None):
        '''
        Parameters
        ----------
        GuassPoints : Number of Guass points, or None to use the element's
                      own QuadOrder.

        Returns
        -------
        gp, gw : Guass points and weights.
        '''

        if GuassPoints is None:

            GuassPoints = self.QuadOrder

        return np.polynomial.legendre.leggauss(GuassPoints)

    def C(self):
        '''
        Returns
        -------
        Cmat : Constitutive relationship matrix. i.e, stress-strain relationship.
        '''

        return ConstitutiveMatrix(self.E, self.v, self.plane, self.LinearFlag)

    def N(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        N : Interpolation Matrix.
        '''

        Nv = self.ShapeFunctions(xi, eta)

        N = np.zeros((2, self.NumDOF))
        N[0, 0::2] = Nv
        N[1, 1::2] = Nv

        return N

    def dN(self, xi, eta):
        '''
        Parameters
        ----------
        xi : local co-ordinate 1.
        eta : local co-ordinate 2.

        Returns
        -------
        dN : Gradient interpolation matrix for the local co-ordinate system.
             Rows are du/dxi, du/deta, dv/dxi, dv/deta.

        '''

        dNv = self.ShapeDerivatives(xi, eta)

        dN = np.zeros((4, self.NumDOF))
        dN[0, 0::2] = dNv[0]
        dN[1, 0::2] = dNv[1]
        dN[2, 1::2] = dNv[0]
        dN[3, 1::2] = dNv[1]

        return dN

    def Map(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        XY : (2x1) Global co-ordinates of the local point.

        '''

        XY = self.N(xi, eta) @ self.NodeCoor.reshape(-1, 1)

        return XY
    
    def Jacobian(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        J : Jacobian Matrix.

        '''

        J = self.ShapeDerivatives(xi, eta) @ self.NodeCoor

        return J

    def InvJ(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        InvJ : Inverse of the Jacobian.

        '''
        InvJ = np.linalg.inv(self.Jacobian(xi, eta))
        
        return InvJ
    
    def detJ(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        detJ : determint of the Jacobian Matrix.
        '''
        
        detJ = np.linalg.det(self.Jacobian(xi, eta))

        return detJ
    
    def B(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        B : Strain Matrix (8x8).
        '''
        
        if self.LinearFlag:
            
            MapJacobian = np.zeros((3,4))
            
            MapJacobian[2,2] = self.InvJ(xi, eta)[0,0]
            MapJacobian[2,3] = self.InvJ(xi, eta)[0,1]
            
        else:
            
             MapJacobian = np.zeros((4,4))
             
             MapJacobian[3,2] = self.InvJ(xi, eta)[0,0]
             MapJacobian[3,3] = self.InvJ(xi, eta)[0,1]
                    
        MapJacobian[0,0] = self.InvJ(xi, eta)[0,0]
        MapJacobian[0,1] = self.InvJ(xi, eta)[0,1]
        
        MapJacobian[1,2] = self.InvJ(xi, eta)[1,0]
        MapJacobian[1,3] = self.InvJ(xi, eta)[1,1]
        
        MapJacobian[2,0] = self.InvJ(xi, eta)[1,0]
        MapJacobian[2,1] = self.InvJ(xi, eta)[1,1]
        
        B = MapJacobian @ self.dN(xi, eta)
        
        return B
    
    def Fvec(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2

        Returns
        -------
        f : Deformation gradient F vector form.

        '''
        I = np.array([[1, 1, 0, 0]]).T
        
        fvec = I + self.B(xi,eta) @ self.U
        
        return fvec
    
    def Fmat(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        fmat : Deformation gradient F matrix form.

        '''
        fvec = self.Fvec(xi, eta)
        
        fmat = np.array([[fvec[0,0], 0, 0.5*fvec[2,0], 0.5*fvec[2,0]],
                         [0, fvec[1,0], 0.5*fvec[3,0], 0.5*fvec[3,0]],
                         [0, fvec[2,0], 0.5*fvec[0,0], 0.5*fvec[0,0]],
                         [fvec[3,0], 0, 0.5*fvec[1,0], 0.5*fvec[1,0]]])
        
        return fmat
    
    def Evec(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        The Green-Lagrane Strain.

        '''
        I = np.array([[1, 1, 0, 0]]).T
        
        evec = 0.5*(self.Fmat(xi, eta).T @ self.Fvec(xi, eta) - I)
        
        return evec
    
    def Svec(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        second Piola-Kirchhoff stress vector.

        '''
        svec = self.C() @ self.Evec(xi, eta)
        
        return svec
    
    
    def StressMat(self, svec):
        '''
        Parameters
        ----------
        svec : second Piola-Kirchhoff stress vector.

        Returns
        -------
        The stress vector arranged as the matrix that pairs with B, so that
        B.T @ StressMat @ B is the geometric stiffness contribution.

        '''
        s = svec

        smat = np.array([[s[0,0], 0, s[2,0], 0],
                         [0, s[1,0], 0, s[2,0]],
                         [s[2,0], 0, s[1,0], 0],
                         [0, s[2,0], 0, s[0,0]]])

        return smat

    def Smat(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        second Piola-Kirchhoff stress matrix.

        '''

        return self.StressMat(self.Svec(xi, eta))
    
    def K(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        Ke : Element Stiffness Matrix as a function of the local co-oridinates.
        
        BT * C * B * detJ
        
        B: Strian Matirx
        C: Constitutive relationship (strain - stress)
        detJ: Relates the area of the element in the local co-ordinates to the global co-oridinates.

        '''
        Ke = self.B(xi, eta).T @ self.C() @ self.B(xi, eta)*self.detJ(xi, eta)
        
        return Ke
    
    def Re(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        Element residual
        
        R = BT * Fmat * Svec * detJ

        '''
        Res = self.B(xi, eta).T @ self.Fmat(xi, eta) @ self.Svec(xi, eta)*self.detJ(xi, eta)
        
        return Res
    
    def KT(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        Tangent Stiffness matrix

        '''
        
        Kt = self.B(xi, eta).T @ (self.Smat(xi, eta) + self.Fmat(xi, eta) @ self.C() @ self.Fmat(xi, eta).T) @ self.B(xi, eta)
        
        return Kt * self.detJ(xi, eta)
    
    def StiffMatrix(self, GuassPoints = None):
        '''
        Parameters
        ----------
        GuassPoints : Select the number of Guass points.
                      The default is 2.

        Returns
        -------
        StiffMatrix : (8x8) Element Stiffness Matrix.
        '''
            
        StiffMatrix = np.zeros((self.NumDOF, self.NumDOF))
        gp, gw = self.GuassPointsAndWeights(GuassPoints) #guass points and weights
        
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                StiffMatrix += self.t * self.K(Xi, Eta) * Wxi * Weta
                     
        return StiffMatrix
    
    def ResTangent(self, GuassPoints = None):
        '''
        Parameters
        ----------
        GuassPoints : Number of Guass Points
            DESCRIPTION. The default is 2.

        Returns
        -------
        TangentMatrix : Integrated Tangent Stiffness Matrix.
        ResidualVector : Integrated residual Vector.

        '''
        TangentMatrix = np.zeros((self.NumDOF, self.NumDOF))
        ResidualVector = np.zeros((self.NumDOF, 1))
        
        gp, gw = self.GuassPointsAndWeights(GuassPoints) #guass points and weights
        
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                TangentMatrix += self.t * self.KT(Xi, Eta) * Wxi * Weta
                
                ResidualVector += self.t * self.Re(Xi, Eta) * Wxi * Weta
    
        return TangentMatrix, ResidualVector
