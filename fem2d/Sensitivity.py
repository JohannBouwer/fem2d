from fem2d.elements4Node import Q4, FiveBeta
from fem2d.elements8Node import Q8
import numpy as np

class dQ4dX(Q4):
    
    def dJdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        Derivitive of the Jacobian w.r.t
        the nodal Coordinates.

        '''
        
        dNdx = np.array([[self.dN1dXi(xi, eta), self.dN2dXi(xi, eta), self.dN3dXi(xi, eta), self.dN4dXi(xi, eta)],
                      [self.dN1dEta(xi, eta), self.dN2dEta(xi, eta), self.dN3dEta(xi, eta), self.dN4dEta(xi, eta)]])
        
        x = np.zeros((4,2)) 
        x[int(DOF//2), DOF%2] = 1
        djdx = dNdx @ x

        return djdx
    
    def dInvJdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        Derivitive of the Inverse of the Jacobian w.r.t
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
        Derivitive of the determint of the Jacobian w.r.t the
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
        Derivitive of the strain matrix w.r.t
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
              the derivitive w.r.t .

        Returns
        -------
        dfvecdx : Derivitive of the derformation gradient.

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
              the derivitive w.r.t .

        Returns
        -------
        dfmatdx : Derivitive of the derformation gradient in matrix form.

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
              the derivitive w.r.t .

        Returns
        -------
        devecdx : Derivitive of the Green-Lagrane strain.

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
              the derivitive w.r.t .

        Returns
        -------
        dsvecdx : derivitive of the second Piola-Kirchhoff stress vector.
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
              the derivitive w.r.t .

        Returns
        -------
        Ke : Derivitive of Element Stiffness Matrix as a function of the local co-oridinates.
        
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
              the derivitive w.r.t .

        Returns
        -------
        drdx : Derivitive of the Element Residual as a functio of the local coordinates.

        '''
        drdx = (np.linalg.multi_dot([self.dBdX(xi, eta, DOF).T, self.Fmat(xi, eta), self.Svec(xi, eta)])*self.detJ(xi, eta) 
             +  np.linalg.multi_dot([self.B(xi, eta).T, self.dFmatdX(xi, eta, DOF), self.Svec(xi, eta)])*self.detJ(xi, eta) 
             +  np.linalg.multi_dot([self.B(xi, eta).T, self.Fmat(xi, eta), self.dSvecdX(xi, eta, DOF)])*self.detJ(xi, eta) 
             +  np.linalg.multi_dot([self.B(xi, eta).T, self.Fmat(xi, eta), self.Svec(xi, eta)])*self.ddetJdX(xi, eta, DOF))
        
        return drdx
    
    def Integrate(self, DOF, GuassPoints = 2):
        '''
        Parameters
        ----------
        DOF : Local degree of freedom to take
              the derivitive w.r.t .
        GuassPoints : TYPE, int
            Number of Gueass Points to use. The default is 2.

        Returns
        -------
        SensMatrix : Derivivite of the element stiffness matrix.

        '''
        SensMatrix = np.zeros((8,8))
        gp, gw = np.polynomial.legendre.leggauss(GuassPoints) #guass points and weights
        
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                SensMatrix += self.t * self.dKdX(Xi, Eta, DOF) * Wxi * Weta
                     
        return SensMatrix
    
    def ResIntegrate(self, DOF, GuassPoints = 2):
        '''
        Parameters
        ----------
        DOF : Local degree of freedom to take
              the derivitive w.r.t.
        GuassPoints : TYPE, int
            Number of Gueass Points to use. The default is 2.

        Returns
        -------
        SensResidual :  Derivivite of the element residual vector..

        '''
        SensResidual = np.zeros((8,1))
        gp, gw = np.polynomial.legendre.leggauss(GuassPoints) #guass points and weights
        
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                SensResidual += self.t * self.dRdX(Xi, Eta, DOF) * Wxi * Weta
                     
        return SensResidual

class dQ8dX(Q8):
    
    def dInvJdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.

        Returns
        -------
        Derivitive of the Inverse of the Jacobian w.r.t
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
        Derivitive of the determint of the Jacobian w.r.t the
        the nodal coordinates.
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
        Derivitive of the strain matrix w.r.t
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
              the derivitive w.r.t .

        Returns
        -------
        dfvecdx : Derivitive of the derformation gradient.

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
              the derivitive w.r.t .

        Returns
        -------
        dfmatdx : Derivitive of the derformation gradient in matrix form.

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
              the derivitive w.r.t .

        Returns
        -------
        devecdx : Derivitive of the Green-Lagrane strain.

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
              the derivitive w.r.t .

        Returns
        -------
        dsvecdx : derivitive of the second Piola-Kirchhoff stress vector.
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
              the derivitive w.r.t .

        Returns
        -------
        Ke : Derivitive of Element Stiffness Matrix as a function of the local co-oridinates.
        
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
              the derivitive w.r.t .

        Returns
        -------
        drdx : Derivitive of the Element Residual as a functio of the local coordinates.

        '''
        drdx = (np.linalg.multi_dot([self.dBdX(xi, eta, DOF).T, self.Fmat(xi, eta), self.Svec(xi, eta)])*self.detJ(xi, eta) 
             +  np.linalg.multi_dot([self.B(xi, eta).T, self.dFmatdX(xi, eta, DOF), self.Svec(xi, eta)])*self.detJ(xi, eta) 
             +  np.linalg.multi_dot([self.B(xi, eta).T, self.Fmat(xi, eta), self.dSvecdX(xi, eta, DOF)])*self.detJ(xi, eta) 
             +  np.linalg.multi_dot([self.B(xi, eta).T, self.Fmat(xi, eta), self.Svec(xi, eta)])*self.ddetJdX(xi, eta, DOF))
        
        return drdx
    
    def dJdX(self, xi, eta, DOF):
        
        dNdx = np.array([[self.dN1dXi(xi, eta), self.dN2dXi(xi, eta), self.dN3dXi(xi, eta), self.dN4dXi(xi, eta), 
                          self.dN5dXi(xi, eta), self.dN6dXi(xi, eta), self.dN7dXi(xi, eta), self.dN8dXi(xi, eta)],
                        [self.dN1dEta(xi, eta), self.dN2dEta(xi, eta), self.dN3dEta(xi, eta), self.dN4dEta(xi, eta),
                         self.dN5dEta(xi, eta), self.dN6dEta(xi, eta), self.dN7dEta(xi, eta), self.dN8dEta(xi, eta)]])
       
        x = np.zeros((8,2)) 
        x[int(DOF//2), DOF%2] = 1
        djdx = dNdx @ x
        
        return djdx
    
    def Integrate(self, DOF, GuassPoints = 3):
        '''
        Parameters
        ----------
        DOF : Local degree of freedom to take
              the derivitive w.r.t .
        GuassPoints : TYPE, int
            Number of Gueass Points to use. The default is 2.

        Returns
        -------
        SensMatrix : Derivivite of the element stiffness matrix.

        '''
        SensMatrix = np.zeros((16,16))
        gp, gw = np.polynomial.legendre.leggauss(GuassPoints) #guass points and weights
        
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                SensMatrix += self.t * self.dKdX(Xi, Eta, DOF) * Wxi * Weta
                     
        return SensMatrix
    
    def ResIntegrate(self, DOF, GuassPoints = 2):
        '''
        Parameters
        ----------
        DOF : Local degree of freedom to take
              the derivitive w.r.t.
        GuassPoints : TYPE, int
            Number of Gueass Points to use. The default is 2.

        Returns
        -------
        SensResidual :  Derivivite of the element residual vector..

        '''
        SensResidual = np.zeros((16,1))
        gp, gw = np.polynomial.legendre.leggauss(GuassPoints) #guass points and weights
        
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                SensResidual += self.t * self.dRdX(Xi, Eta, DOF) * Wxi * Weta
                     
        return SensResidual
    

class d5BdX(dQ4dX, FiveBeta):
    
    def dPdX(self, xi, eta, DOF):
        '''
        Parameters
        ----------
        xi : Local variable 1.
        eta : Local variable 2.
        DOF : Local degree of freedom to take
              the derivitive w.r.t.

        Returns
        -------
        dPdX : derivitive of the Interpolation Matrix for the assumend Stress Element.
        '''
        
        mat = 1/4*np.array([[-1, 1, 1, -1],
                             [1, -1, 1, -1],
                             [-1, -1, 1, 1]])
        
        A = mat @ self.NodeCoor[:,[0]]
        B = mat @ self.NodeCoor[:,[1]]
        
        a1, a3 = A[0,0], A[2, 0]
        b1, b3 = B[0, 0], B[2, 0] 

        if DOF%2 == 0:
            
            dX = np.zeros((4,1))
            dX[DOF//2, 0] = 1
            dA = mat @ dX
            
            da1, da3 = dA[0,0], dA[2, 0]
            
            dP = np.array([[0, 0, 0, 2*a1*da1*xi, 2*a3*da3*eta],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, b1*da1*xi, b3*da3*eta]])
           
            
        else:
            
            dX = np.zeros((4,1))
            dX[DOF//2, 0] = 1
            dB = mat @ dX
            
            db1, db3 = dB[0,0], dB[2, 0]
            
            dP = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 2*b1*db1*xi, 2*b3*db3*eta],
                           [0, 0, 0, a1*db1*xi, a3*db3*eta]])
            
            
        if self.LinearFlag:
            
            dPdX = dP
            
        else:
            
            dPdX = np.zeros((4, 5))
            
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
              the derivitive w.r.t.

        Returns
        -------
        dHdX : derivitive of the local varible He for the assumend stress element.
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
              the derivitive w.r.t.

        Returns
        -------
        dMdX : derivitive of the local varible Me for the assumend stress element.
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
              the derivitive w.r.t.

        Returns
        -------
        dGdX : derivitive of the local varible Ge for the assumend stress element.
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
    
    def Integrate(self, DOF, GuassPoints = 2): 
        '''
        Parameters
        ----------
        DOF : Local degree of freedom to take
              the derivitive w.r.t .
        GuassPoints : TYPE, int
            Number of Gueass Points to use. The default is 2.
    
        Returns
        -------
        SensMatrix : Derivivite of the element stiffness matrix.
    
        '''
        
        G = np.zeros((8,5))
        H = np.zeros((5,5))
        
        dG = np.zeros((8,5))
        dH = np.zeros((5,5))
            
        gp, gw = np.polynomial.legendre.leggauss(GuassPoints) #guass points and weights
    
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                G += self.Ge(Xi, Eta) * Wxi * Weta
                
                H += self.He(Xi, Eta) * Wxi * Weta
                
                dH += self.dHedX(Xi, Eta, DOF) * Wxi * Weta
                
                dG += self.dGedX(Xi, Eta, DOF) * Wxi * Weta
                
        
        SensStiffMatrix = self.t * (dG @ np.linalg.inv(H) @ G.T
                                    + G @ (-1*np.linalg.inv(H) @ dH @ np.linalg.inv(H)) @ G.T
                                    + G @ np.linalg.inv(H) @ dG.T)
        
        return SensStiffMatrix

    def ResIntegrate(self, DOF, GuassPoints = 2):
        '''
        Parameters
        ----------
        DOF : Local degree of freedom to take
              the derivitive w.r.t.
        GuassPoints : TYPE, int
            Number of Gueass Points to use. The default is 2.

        Returns
        -------
        SensResidual :  Derivivite of the element residual vector..

        '''
        SensResidual = np.zeros((8,1))
        gp, gw = np.polynomial.legendre.leggauss(GuassPoints) #guass points and weights
        
        dG = np.zeros((8,5))
        dH = np.zeros((5,5))
        dM = np.zeros((5,1))
        
        G = np.zeros((8,5))
        H = np.zeros((5,5))
        M = np.zeros((5,1))
        
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                dM += self.dMedX(Xi, Eta, DOF) * Wxi * Weta
                
                dH += self.dHedX(Xi, Eta, DOF) * Wxi * Weta
                
                H += self.He(Xi, Eta) * Wxi * Weta
                
                M += self.Me(Xi, Eta) * Wxi * Weta
        
        B = np.linalg.inv(H) @ M
        dB = (-1*np.linalg.inv(H) @ dH @ np.linalg.inv(H)) @ M + np.linalg.inv(H) @ dM
        
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                G += self.Ge(Xi, Eta) * Wxi * Weta
                
                dG += self.dGedX(Xi, Eta, DOF) * Wxi * Weta
 
        SensResidual = self.t*dG @ B + self.t*G @ dB
        
        return SensResidual
    
            
    
    