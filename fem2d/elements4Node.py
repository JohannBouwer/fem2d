import numpy as np

class Q4(object):
    
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
        
        #Shape Functions
        self.N1 = lambda xi, eta : 1/4 * (1 - xi)*(1 - eta)
        self.N2 = lambda xi, eta : 1/4 * (1 + xi)*(1 - eta)
        self.N3 = lambda xi, eta : 1/4 * (1 + xi)*(1 + eta)
        self.N4 = lambda xi, eta : 1/4 * (1 - xi)*(1 + eta)
        
        #Shape Function Derivitives
        self.dN1dXi = lambda xi, eta : -1/4 * (1 - xi)
        self.dN2dXi = lambda xi, eta : 1/4 * (1 - xi)
        self.dN3dXi = lambda xi, eta : 1/4 * (1 + xi)
        self.dN4dXi = lambda xi, eta : -1/4 * (1 + xi)
        
        self.dN1dEta = lambda xi, eta : -1/4 * (1 - eta)
        self.dN2dEta = lambda xi, eta : -1/4 * (1 + eta)
        self.dN3dEta = lambda xi, eta : 1/4 * (1 + eta)
        self.dN4dEta = lambda xi, eta : 1/4 * (1 - eta)
        
        return 
    
    def C(self):
        '''
        Returns
        -------
        Cmat : Constitutive relationship matrix. i.e, stress-strain relationship.
        '''

        if self.LinearFlag:
            
            if self.plane%2 == 0: #Plane Stress
                
                Cmat = self.E/(1 - self.v**2) * np.array([[1, self.v, 0],
                                                          [self.v, 1, 0],
                                                          [0, 0, (1 - self.v)/2]])
            else: #Plane Strain
                
                Cmat = self.E/((1 + self.v)*(1 - 2*self.v)) * np.array([[1 - self.v, self.v, 0],
                                                                        [self.v, 1 - self.v, 0],
                                                                        [0, 0, (1 - 2*self.v)/2]])
        
        
        else:
            
            if self.plane%2 == 0:
                
                Cmat = self.E/(1 - self.v**2) * np.array([[1, self.v, 0, 0],
                                                          [self.v, 1, 0, 0],
                                                          [0, 0, 1 - self.v, 0],
                                                          [0, 0, 0, 1 - self.v]])
                
            else:
                
                E = self.E/(1 - self.v**2)
                v = self.v/(1 - self.v)
                
                Cmat = E/(1 - v**2) * np.array([[1, v, 0, 0],
                                                [v, 1, 0, 0],
                                                [0, 0, 1 - v, 0],
                                                [0, 0, 0, 1 - v]])
        
        return Cmat
    
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
        
        N = np.array([[self.N1(xi, eta), 0, self.N2(xi, eta), 0, self.N3(xi, eta), 0, self.N4(xi, eta), 0],
                      [0, self.N1(xi, eta), 0, self.N2(xi, eta), 0, self.N3(xi, eta), 0, self.N4(xi, eta)]])
        
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

        '''
        
        dN = np.array([[self.dN1dXi(xi, eta), 0, self.dN2dXi(xi, eta), 0, self.dN3dXi(xi, eta), 0, self.dN4dXi(xi, eta), 0],
                       [self.dN1dEta(xi, eta), 0, self.dN2dEta(xi, eta), 0, self.dN3dEta(xi, eta), 0, self.dN4dEta(xi, eta), 0],
                       [0, self.dN1dXi(xi, eta), 0, self.dN2dXi(xi, eta), 0, self.dN3dXi(xi, eta), 0, self.dN4dXi(xi, eta)],
                       [0, self.dN1dEta(xi, eta), 0, self.dN2dEta(xi, eta), 0, self.dN3dEta(xi, eta), 0, self.dN4dEta(xi, eta)]])
        
        return dN
    
    def Map(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        XY : Global co-ordiantes from local co-ordinates in an element.

        '''
        
        XY = self.N(xi, eta) @ self.NodeCoor
        
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
        
        dNdx = np.array([[self.dN1dXi(xi, eta), self.dN2dXi(xi, eta), self.dN3dXi(xi, eta), self.dN4dXi(xi, eta)],
                      [self.dN1dEta(xi, eta), self.dN2dEta(xi, eta), self.dN3dEta(xi, eta), self.dN4dEta(xi, eta)]])

        J = dNdx @ self.NodeCoor

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
        s = self.Svec(xi, eta)
        
        smat = np.array([[s[0,0], 0, s[2,0], 0],
                         [0, s[1,0], 0, s[2,0]],
                         [s[2,0], 0, s[1,0], 0],
                         [0, s[2,0], 0, s[0,0]]])
        
        return smat
    
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
    
    def StiffMatrix(self, GuassPoints = 2):
        '''
        Parameters
        ----------
        GuassPoints : Select the number of Guass points.
                      The default is 2.

        Returns
        -------
        StiffMatrix : (8x8) Element Stiffness Matrix.
        '''
            
        StiffMatrix = np.zeros((8,8))
        gp, gw = np.polynomial.legendre.leggauss(GuassPoints) #guass points and weights
        
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                StiffMatrix += self.t * self.K(Xi, Eta) * Wxi * Weta
                     
        return StiffMatrix
    
    def ResTangent(self, GuassPoints = 2):
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
        TangentMatrix = np.zeros((8,8))
        ResidualVector = np.zeros((8,1))
        
        gp, gw = np.polynomial.legendre.leggauss(GuassPoints) #guass points and weights
        
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                TangentMatrix += self.t * self.KT(Xi, Eta) * Wxi * Weta
                
                ResidualVector += self.t * self.Re(Xi, Eta) * Wxi * Weta
    
        return TangentMatrix, ResidualVector
    

class FiveBeta(Q4):
    
    def P(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        P : Interpolation Matris for the assumend Stress Element.
        '''
        mat = 1/4*np.array([[-1, 1, 1, -1],
                             [1, -1, 1, -1],
                             [-1, -1, 1, 1]])
        
        A = mat @ self.NodeCoor[:,[0]]
        B = mat @ self.NodeCoor[:,[1]]
        
        a1, a3 = A[0,0], A[2, 0]
        b1, b3 = B[0, 0], B[2, 0] 
        
        if self.LinearFlag:
            
            P = np.array([[1, 0, 0, a1**2*xi, a3**2*eta],
                          [0, 1, 0, b1**2*xi, b3**2*eta],
                          [0, 0, 1, a1*b1*xi, a3*b3*eta]])
            
        else:
            
             P = np.array([[1, 0, 0, a1**2*xi, a3**2*eta],
                          [0, 1, 0, b1**2*xi, b3**2*eta],
                          [0, 0, 1, a1*b1*xi, a3*b3*eta],
                          [0, 0, 1, a1*b1*xi, a3*b3*eta]])
             
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
    
    def Le(self, xi, eta):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        L : Variable needed for Residual and Tangent.

        '''
        L = self.B(xi, eta).T @ self.Smat(xi, eta) @ self.B(xi, eta) * self.detJ(xi, eta)
        
        return L
    
    def StiffMatrix(self, GuassPoints = 2):
        '''
        Parameters
        ----------
        GuassPoints : Select the number of Guass points.
                      The default is 2.

        Returns
        -------
        StiffMatrix : (8x8) Element Stiffness Matrix.
        '''
        
        G = np.zeros((8,5))
        H = np.zeros((5,5))
            
        gp, gw = np.polynomial.legendre.leggauss(GuassPoints) #guass points and weights
    
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                G += self.Ge(Xi, Eta) * Wxi * Weta
                
                H += self.He(Xi, Eta) * Wxi * Weta
        
        StiffMatrix = self.t * G @ np.linalg.inv(H) @ G.T
        
        return StiffMatrix
    
    def ResTangent(self, GuassPoints = 2):
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
        G = np.zeros((8,5))
        H = np.zeros((5,5))
        M = np.zeros((5,1))
        L = np.zeros((8,8))
            
        gp, gw = np.polynomial.legendre.leggauss(GuassPoints) #guass points and weights
    
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                M += self.Me(Xi, Eta) * Wxi * Weta
                
                H += self.He(Xi, Eta) * Wxi * Weta
        
        B = np.linalg.inv(H) @ M
        
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                G += self.Ge(Xi, Eta) * Wxi * Weta
                
                L += self.Le(Xi, Eta) * Wxi * Weta

        ResidualVector = self.t * G @ B
       
        TangentMatrix = self.t * (L + G @ np.linalg.inv(H) @ G.T)
        
        return TangentMatrix, ResidualVector
        
        
        
        
        
