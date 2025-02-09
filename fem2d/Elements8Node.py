import numpy as np
from fem2d.elements4Node import Q4
from numpy.polynomial.legendre import leggauss

class Q8(Q4): #inherit unspecified functions from Q4 class
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
        
        #shape functions
        self.N1 = lambda xi, eta: -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta)
        self.N2 = lambda xi, eta: -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta)
        self.N3 = lambda xi, eta: -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta)
        self.N4 = lambda xi, eta: -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta)
        self.N5 = lambda xi, eta: 0.5 * (1 - xi**2) * (1 - eta)
        self.N6 = lambda xi, eta: 0.5 * (1 + xi) * (1 - eta**2)
        self.N7 = lambda xi, eta: 0.5 * (1 - xi**2) * (1 + eta)
        self.N8 = lambda xi, eta: 0.5 * (1 - xi) * (1 - eta**2)
        
        #shape function derivitives
        self.dN1dXi = lambda xi, eta : -0.25 * (eta - 1)*(eta + 2*xi)
        self.dN2dXi = lambda xi, eta : -0.25 * (-eta + 2*xi)*(eta - 1)
        self.dN3dXi = lambda xi, eta : -0.25 * (-eta - 2*xi)*(eta + 1)
        self.dN4dXi = lambda xi, eta : -0.25 * (eta + 1)*(eta - 2*xi)
        self.dN5dXi = lambda xi, eta : 0.5 * 2*xi*(eta - 1)
        self.dN6dXi = lambda xi, eta : 0.5 * (1 - eta**2)
        self.dN7dXi = lambda xi, eta : 0.5 * -2*xi*(eta + 1)
        self.dN8dXi = lambda xi, eta : 0.5 * (eta**2 - 1)
        
        self.dN1dEta = lambda xi, eta : -0.25 * (2*eta + xi)*(xi - 1)
        self.dN2dEta = lambda xi, eta : -0.25 * (-2*eta + xi)*(xi + 1)
        self.dN3dEta = lambda xi, eta : -0.25 * (-2*eta - xi)*(xi + 1)
        self.dN4dEta = lambda xi, eta : -0.25 * (2*eta - xi)*(xi - 1)
        self.dN5dEta = lambda xi, eta : 0.5 * (xi**2 - 1)
        self.dN6dEta = lambda xi, eta : 0.5 * -2*eta*(xi + 1)
        self.dN7dEta = lambda xi, eta : 0.5 * (1 - xi**2)
        self.dN8dEta = lambda xi, eta : 0.5 * 2*eta*(xi - 1)
    
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
        N = np.array([[self.N1(xi, eta), 0 , self.N2(xi, eta), 0, self.N3(xi, eta), 0, self.N4(xi, eta), 0, 
                       self.N5(xi, eta), 0, self.N6(xi, eta), 0, self.N7(xi, eta), 0, self.N8(xi, eta), 0],
                      [0, self.N1(xi, eta), 0 , self.N2(xi, eta), 0, self.N3(xi, eta), 0, self.N4(xi, eta), 
                       0, self.N5(xi, eta), 0, self.N6(xi, eta), 0, self.N7(xi, eta), 0, self.N8(xi, eta)]])
        
        return N
    
    def dN(self, xi, eta):
        '''
        Parameters
        ----------
        xi : local co-ordinate 1.
        eta : local co-ordinate 2.

        Returns
        -------
        dN : Gradient interpolation matrix for the local co-ordinate system. (4 x 16)

        '''
        
        dN = np.array([[self.dN1dXi(xi, eta), 0, self.dN2dXi(xi, eta), 0, self.dN3dXi(xi, eta), 0, self.dN4dXi(xi, eta), 0, 
                        self.dN5dXi(xi, eta), 0, self.dN6dXi(xi, eta), 0, self.dN7dXi(xi, eta), 0, self.dN8dXi(xi, eta), 0],
                       [self.dN1dEta(xi, eta), 0, self.dN2dEta(xi, eta), 0, self.dN3dEta(xi, eta), 0, self.dN4dEta(xi, eta), 0, 
                        self.dN5dEta(xi, eta), 0, self.dN6dEta(xi, eta), 0, self.dN7dEta(xi, eta), 0, self.dN8dEta(xi, eta), 0],
                       [0, self.dN1dXi(xi, eta), 0, self.dN2dXi(xi, eta), 0, self.dN3dXi(xi, eta), 0, self.dN4dXi(xi, eta), 0,
                        self.dN5dXi(xi, eta), 0, self.dN6dXi(xi, eta), 0, self.dN7dXi(xi, eta), 0, self.dN8dXi(xi, eta)],
                       [0, self.dN1dEta(xi, eta), 0, self.dN2dEta(xi, eta), 0, self.dN3dEta(xi, eta), 0, self.dN4dEta(xi, eta), 0, 
                        self.dN5dEta(xi, eta), 0, self.dN6dEta(xi, eta), 0, self.dN7dEta(xi, eta), 0, self.dN8dEta(xi, eta)]])
        
        return dN
    
    def Map(self, xi, eta):
        '''
        
        FIX THE SHAPES
        
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
        
        dNdx = np.array([[self.dN1dXi(xi, eta), self.dN2dXi(xi, eta), self.dN3dXi(xi, eta), self.dN4dXi(xi, eta), 
                          self.dN5dXi(xi, eta), self.dN6dXi(xi, eta), self.dN7dXi(xi, eta), self.dN8dXi(xi, eta)],
                        [self.dN1dEta(xi, eta), self.dN2dEta(xi, eta), self.dN3dEta(xi, eta), self.dN4dEta(xi, eta),
                         self.dN5dEta(xi, eta), self.dN6dEta(xi, eta), self.dN7dEta(xi, eta), self.dN8dEta(xi, eta)]])
        
        J = dNdx @ self.NodeCoor

        return J

    def StiffMatrix(self, GuassPoints = 3):
        '''
        Parameters
        ----------
        GuassPoints : Select the number of Guass points.
                      The default is 3.

        Returns
        -------
        StiffMatrix : (8x8) Element Stiffness Matrix.
        '''
            
        StiffMatrix = np.zeros((16,16))
        gp, gw = np.polynomial.legendre.leggauss(GuassPoints) #guass points and weights
        
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                StiffMatrix += self.t * self.K(Xi, Eta) * Wxi * Weta
                     
        return StiffMatrix
        
    
    def ResTangent(self, GuassPoints = 3):
        '''
        Parameters
        ----------
        GuassPoints : Number of Guass Points
            DESCRIPTION. The default is 3.

        Returns
        -------
        TangentMatrix : Integrated Tangent Stiffness Matrix.
        ResidualVector : Integrated residual Vector.

        '''
        TangentMatrix = np.zeros((16,16))
        ResidualVector = np.zeros((16,1))
        
        gp, gw = np.polynomial.legendre.leggauss(GuassPoints) #guass points and weights
        
        for Xi, Wxi in zip(gp, gw):
            
            for Eta, Weta in zip(gp, gw):
                
                TangentMatrix += self.t * self.KT(Xi, Eta) * Wxi * Weta
                
                ResidualVector += self.t * self.Re(Xi, Eta) * Wxi * Weta
    
        return TangentMatrix, ResidualVector   



  
