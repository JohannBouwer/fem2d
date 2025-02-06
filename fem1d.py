import numpy as np
import matplotlib.pyplot as plt

class Q3(object):
    
    def __init__(self, A, E, NodeCoor):
        '''
        Parameters
        ----------
        A : Area (ethier constant or a function of global domain)
        E : Youngs Modulous (ethier constant or a function of global domain)
        NodeCoor : Global Nodal co-ordinates

        Description
        -------
        A 1D beam element using quadratic shape functions and 
        2 points guass quadrature.
        '''
        
        self.NodeCoor = NodeCoor
        
        if type(E) == float:
            self.E = lambda x : E
        else:
            self.E = E
        
        if type(A) == float:
            self.A = lambda x : A
        else:
            self.A = A
        
        #Define Quadratic Shape Functions
        
        self.N1 = lambda Xi: -1/2*Xi*(1 - Xi)
        self.N2 = lambda Xi:  1/2*Xi*(1 + Xi)
        self.N3 = lambda Xi: (Xi + 1)*(1 - Xi)
        
        #Gradients of Shape Funcitons
        
        self.dN1dXi = lambda Xi: Xi - 1/2
        self.dN2dXi = lambda Xi: Xi + 1/2
        self.dN3dXi = lambda Xi: -2*Xi
        
    def X_map(self, Xi):
        '''
        Parameters
        ----------
        Xi : Local Co-ordinate system.

        Returns
        -------
        Maps the local co-ordinate for an element to the global co-ordinate.

        '''
        
        return np.array([[self.N1(Xi), self.N2(Xi), self.N3(Xi)]]).dot(self.NodeCoor)
    
    def dUdXi(self, Xi):
        '''
        Parameters
        ----------
        Xi : Local Co-ordinate system.

        Returns
        -------
        Gradient vector of the shape functions.
        '''
        
        return np.array([[self.dN1dXi(Xi), self.dN2dXi(Xi), self.dN3dXi(Xi)]])
    
    def dXdXi(self, Xi):
        '''

        Parameters
        ----------
        Xi : Local co-ordinate system.

        Returns
        -------
        Known as the Jacobian, how a change in local system changes in
        the global system.
        '''
        
        return self.dUdXi(Xi).dot(self.NodeCoor)
    
    def dXidX(self, Xi):
        '''
        Parameters
        ----------
        Xi : Local co-ordinate system.

        Returns
        -------
        The inverse of the Jocobain, how a change in gloabl system changes in
        the local system. .
        '''
        return 1/self.dXdXi(Xi)
    
    def B(self, Xi):
        '''
        Parameters
        ----------
        Xi : Local co-ordinate system.

        Returns
        -------
        The strain matrix B. Effectly dUdX, found using the chain rule.
        '''

        return self.dXidX(Xi)*self.dUdXi(Xi)
    
    def FullStffness(self, Xi):
        '''
        Parameters
        ----------
        Xi : The local co-ordinate system.

        Returns
        -------
        The matrix that will be integrated to find the element stiffness matrix (3x3).
        
        BT E B A dXdXi:
            -BT Transpose of the strian matrix, found from the weighted resudial method as
            we assume weight functions same shape functions.
            -E Youngs Modulous.
            -B The strain matrix.
            -A The area of the beam.
            -dXdXi The mapping from global to local systems needed so that Guass Quad can be used.
        '''
        
        x = self.X_map(Xi)

        return self.B(Xi).T*self.E(x)*self.B(Xi)*self.A(x)*self.dXdXi(Xi)
    
    def Integrate(self):
        '''
        Returns
        -------
        Complets a 2 point Guass Quadrature integration to find the 
        element stiffness matrix.
        '''
        
        return self.FullStffness(-1/3**0.5) + self.FullStffness(1/3**0.5)
    
    
class FEM_Solver(object):
    
    def __init__(self, Mesh):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the Mesher class.

        Description
        -------
        Solves the displacement field.
        '''
        
        self.Mesh = Mesh
    
    def Assemble(self):
        '''
        Returns
        -------
        K : Assembles the global stiffness matrix using the problem defined
        in the Mesh object.
        '''
        
        K = np.zeros((self.Mesh.Nodes.shape[0], self.Mesh.Nodes.shape[0]))
        
        for el in range(self.Mesh.Elements.shape[0]):
            
            # Node numbers in the element
            
            LocalNode1 = self.Mesh.Elements[el, 1] - 1 
            LocalNode2 = self.Mesh.Elements[el, 2] - 1
            LocalNode3 = self.Mesh.Elements[el, 3] - 1
            
            KCoor = [LocalNode1, LocalNode2, LocalNode3]
            
            #Node gloabl coordinates
            
            NodeCoor = self.Mesh.Nodes[self.Mesh.Elements[el, 1:]-1, 1].reshape(-1,1) 

            K[np.ix_(KCoor, KCoor)] += Q3(self.Mesh.A, self.Mesh.E, NodeCoor).Integrate()
            
        return K
    
    def LinearSolve(self, boundary, Load):
        '''
        Parameters
        ----------
        boundary : The fixed nodes in the mesh. TYPE np.array
        Load : Magnitude and location of the load. [Magnitude, Node]

        Returns
        -------
        The displacemnt of the Nodes on the Mesh.
        '''
        
        self.Mesh.U = np.zeros((self.Mesh.Nodes.shape[0], 1))
        
        K = self.Assemble()
        
        self.Mesh.Kpp = K[np.ix_(boundary - 1, boundary - 1)]
        
        FreeNodes = self.Mesh.Nodes[:,0].astype('int') - 1
        
        for i in boundary:
            FreeNodes = FreeNodes[FreeNodes != i - 1]
        
        self.Mesh.Kff = K[np.ix_(FreeNodes, FreeNodes)]

        F = np.zeros((self.Mesh.Nodes.shape[0],1))
        F[Load[:,0].astype('int') - 1, 0] = Load[:,1]
        
        self.Mesh.Uff = np.linalg.solve(self.Mesh.Kff, F[FreeNodes,:])
        
        self.Mesh.U[boundary - 1,0] = 0
        self.Mesh.U[FreeNodes, 0] = self.Mesh.Uff[:,0]
        
        return self.Mesh.U
    
class Mesher(object):
     
    def Beam(self, A, L, E, elnum):
        '''
        Parameters
        ----------
        A : Area (ethier constant or a function of global domain)
        L : Length.
        E : Youngs Modulous (ethier constant or a function of global domain)
        elnum : Number of Elements.

        Returns
        -------
        Matrix of the Elements:
        1 1 3 2: denotes Element 1 is made with nodes 1 3 2.
        Matrix of Nodes:
            1 0
            2 0.5
        denotes the node number and the global location of
        the node.
        '''
        
        if type(E) == float:
            self.E = lambda x : E
        else:
            self.E = E
        
        if type(A) == float:
            self.A = lambda x : A
        else:
            self.A = A
        
        el_Length = L/elnum
        
        self.Elements = np.zeros((elnum, 4))
        self.Elements[:,0] = np.arange(1, elnum+1, 1)
        
        for i in range(elnum):
            
            self.Elements[i,1:] = np.array([1, 3, 2]) + i*2
        
        self.Elements = self.Elements.astype('int')
        
        self.Nodes = np.zeros((2*elnum + 1, 2))
        self.Nodes[:,0] = np.arange(1, 2*elnum + 2, 1)
        
        for i in range(2*elnum + 1):
            
            self.Nodes[i,1] = i*el_Length/2
        
        return self.Elements, self.Nodes
    
class PostProcessing(object):
    
    def InitialMesh(Mesh, title = 'Mesh'):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the mesher class.
        title : str, title of the figure
            DESCRIPTION. The default is 'Deformation'.

        Returns
        -------
        Creates a plot of the intial mesh.
        fig : figure for user editing.
        ax : axes for extra plotting or editing by user.
        '''
        
        fig = plt.figure('Inital Mesh')
        ax = fig.add_subplot(111)
        
        ax.plot(Mesh.Nodes[:,1], np.zeros_like(Mesh.Nodes[:,1]), 'k', marker = '*')

        return fig, ax
    
    def Deformation(Mesh, title = 'Deformation'):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the mesher class.
        title : str, title of the figure
            DESCRIPTION. The default is 'Deformation'.

        Returns
        -------
        Creates a plot of the deformend mesh.
        fig : figure for user editing.
        ax : axes for extra plotting or editing by user.
        '''
        
        fig = plt.figure(title)
        ax = fig.add_subplot(111)
        
        ax.plot(Mesh.Nodes[:,1] + Mesh.U[:,0], np.zeros_like(Mesh.Nodes[:,1]), 'k', marker = '*')

        return fig, ax
    
    def Overlay(Mesh):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the mesher class.
        title : str, title of the figure
            DESCRIPTION. The default is 'Deformation'.

        Returns
        -------
        Creates an overlay plot of the undeformend and deformend mesh.
        fig : figure for user editing.
        ax : axes for extra plotting or editing by user.
        '''
        
        fig, ax = PostProcessing.Deformation(Mesh, title = 'Overlay')
        
        ax.plot(Mesh.Nodes[:,1], np.zeros_like(Mesh.Nodes[:,1]), 'r', marker = '*')
        
        return fig, ax
    
    def Displacement(Mesh):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the mesher class..

        Returns
        -------
        Creates an result plot of the displcament through the structure.
        fig : figure for user editing.
        ax : axes for extra plotting or editing by user.

        '''
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        for el in range(Mesh.Elements.shape[0]):
            # Orginal Node gloabl coordinates
            
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, [1]] 
            
            #Displcement of Nodes
            NodeCoorDisp = Mesh.U[Mesh.Elements[el, 1:]-1, [0]]
            
            Element = Q3(Mesh.A, Mesh.E, NodeCoor)
            ElementDeformend = Q3(Mesh.A, Mesh.E, NodeCoorDisp)
            
            ElementDisp = np.zeros(10)
            GlobalCoor = np.zeros(10)
            for i, xi in enumerate(np.linspace(-1,1,10)):
                
                ElementDisp[i] = ElementDeformend.X_map(xi)
                GlobalCoor[i] = Element.X_map(xi)
               
            ax.plot(GlobalCoor, ElementDisp)
            
        ax.set(ylabel = 'Displacement M', xlabel = 'Domian M')
        fig.tight_layout()
        
        return fig, ax
    
    def Strain(Mesh):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the mesher class..

        Returns
        -------
        Creates an result plot of the strain through the structure.
        fig : figure for user editing.
        ax : axes for extra plotting or editing by user.

        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        for el in range(Mesh.Elements.shape[0]):
            # Orginal Node gloabl coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, [1]]
            
            #Displcement of Nodes
            NodeCoorDisp = Mesh.U[Mesh.Elements[el, 1:]-1, [0]]
            
            Element = Q3(Mesh.A, Mesh.E, NodeCoor)
            
            ElementStress = np.zeros(100)
            GlobalCoor = np.zeros(100)
            for i, xi in enumerate(np.linspace(-1,1,100)):
                
                GlobalCoor[i] = Element.X_map(xi)
                ElementStress[i] = Element.B(xi).dot(NodeCoorDisp)
               
            ax.plot(GlobalCoor, ElementStress)
            
        ax.set(ylabel = 'Strain', xlabel = 'Domian M')
        fig.tight_layout()
        
        return fig, ax
    
    def Stress(Mesh):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the mesher class..

        Returns
        -------
        Creates an result plot of the stress through the structure.
        fig : figure for user editing.
        ax : axes for extra plotting or editing by user.
        '''
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        for el in range(Mesh.Elements.shape[0]):
            # Orginal Node gloabl coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, [1]] 
            
            #Displcement of Nodes
            NodeCoorDisp = Mesh.U[Mesh.Elements[el, 1:]-1, [0]]

            Element = Q3(Mesh.A, Mesh.E, NodeCoor)

            ElementStress = np.zeros(1000)
            GlobalCoor = np.zeros(1000)
            for i, xi in enumerate(np.linspace(-1,1,1000)):
                
                GlobalCoor[i] = Element.X_map(xi)
                ElementStress[i] = Mesh.E(GlobalCoor[i])*Element.B(xi).dot(NodeCoorDisp)
               
            ax.plot(GlobalCoor, ElementStress/1e6 )
            
        ax.set(ylabel = 'Stress MPa', xlabel = 'Domian M')
        fig.tight_layout()
        
        return fig, ax
    
    def InternalForce(Mesh):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the mesher class..

        Returns
        -------
        Creates an result plot of the stress through the structure.
        fig : figure for user editing.
        ax : axes for extra plotting or editing by user.
        '''
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        for el in range(Mesh.Elements.shape[0]):
            # Orginal Node gloabl coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, [1]]
            
            #Displcement of Nodes
            NodeCoorDisp = Mesh.U[Mesh.Elements[el, 1:]-1, [0]]  
            
            Element = Q3(Mesh.A, Mesh.E, NodeCoor)

            ElementForce = np.zeros(1000)
            GlobalCoor = np.zeros(1000)
            for i, xi in enumerate(np.linspace(-1,1,1000)):
                
                GlobalCoor[i] = Element.X_map(xi) 
                ElementForce[i] = Mesh.E(GlobalCoor[i])*Element.B(xi).dot(NodeCoorDisp) * Mesh.A(Element.X_map(xi))
               
            ax.plot(GlobalCoor, ElementForce)
            
        ax.set(ylabel = 'Force N', xlabel = 'Domian M')
        fig.tight_layout()
        
        return fig, ax
        
        
    

        
        
    
    
        
        
    
    
    
    
        