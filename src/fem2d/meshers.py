import numpy as np

from fem2d.materials import PlaneState


# node*2 - 2 : x direction
# node*2 - 1 : y direction
class Mesh:

    def __init__(self, E, v, thickness, plane, ElementType):
        '''
        Parameters
        ----------
        E : Youngs modulus.
        v : poission ratio.
        thickness : thickness in the 3rd dimensions.
        plane : plane strain or plane stress.
                option 1 or 2.
        ElementType : Type of element.
                    option: q4, 5B.

        Returns
        -------
        None.

        '''
        self.E = E
        self.v = v
        self.t = thickness
        self.plane = PlaneState.From(plane)

        self.ElementType = ElementType

        return

    def _NodeSensitivity(self, Rebuild, Variables, step = 1e-6):
        '''
        Parameters
        ----------
        Rebuild : Callable taking a list of design variables and returning the
                  Nodes array the mesher would build from them.
        Variables : Current values of the design variables.
        step : Forward difference step on each variable. The default is 1e-6.

        Returns
        -------
        dXdx : (NumNodes, 2*NumVariables) derivative of every nodal co-ordinate
               w.r.t each design variable, laid out as the solvers expect.

        '''
        Base = Rebuild(list(Variables))

        dXdx = np.zeros((Base.shape[0], 2*len(Variables)))

        for i in range(len(Variables)):

            Perturbed = list(Variables)
            Perturbed[i] += step

            dXdx[:, 2*i : 2*i + 2] = (Rebuild(Perturbed)[:, 1:] - Base[:, 1:])/step

        return dXdx

    def _LeeFrameNodes(self, el_num, var):
        '''
        Parameters
        ----------
        el_num : Number of elements.
        var : array of the two member lengths.

        Returns
        -------
        Nodes : Node array for the Lee frame with these member lengths.

        '''
        length_up = var[0]
        length_side = var[1]
        t = 2
        Nodes = np.zeros((2*el_num+2,3))

        up_nodes = np.linspace(0,length_up-t,int((el_num-1)/2)+1)
        up_nodes = np.hstack((up_nodes,np.array([length_up])))

        side_nodes = np.linspace(t,length_side,int(el_num/2)+1)

        Nodes[:,0] = np.arange(1,2*el_num+3)

        Nodes[1:el_num + 2:2,1] = np.ones(len(up_nodes))*t
        Nodes[1:el_num + 2:2,2] = up_nodes
        Nodes[:el_num + 2:2,2] = up_nodes

        Nodes[el_num+2::2,2] = np.ones(len(side_nodes[1:]))*length_up
        Nodes[el_num+3::2,2] = np.ones(len(side_nodes[1:]))*length_up-t
        Nodes[el_num+2::2,1] = side_nodes[1:]
        Nodes[el_num + 3::2,1] = side_nodes[1:]

        return Nodes

    #: Included angle of the arch in degrees. 215 is the DaDeppo and Schmidt
    #: deep circular arch, which is the geometry the shape design work uses.
    ArchSpan = 215.0

    def _ArchAngles(self, Number, Span = None):
        '''
        Parameters
        ----------
        Number : How many angles to return.
        Span : Included angle in degrees, or None for ArchSpan.

        Returns
        -------
        Number angles in radians, evenly spaced over the span and centred on the
        crown at 90 degrees.

        '''
        if Span is None:

            Span = self.ArchSpan

        return np.deg2rad(np.linspace(90 - Span/2, 90 + Span/2, Number))

    def _RadiusSpline(self, theta, r):
        '''
        Parameters
        ----------
        theta : Angles of the design points.
        r : Radii at those angles.

        Returns
        -------
        Interpolant for the radius at any angle. Cubic, which is what the shape
        design work uses, dropping to the highest order the point count allows:
        cubic needs four points, so a single design variable, which gives only
        three points with the two fixed ends, falls back to quadratic.

        '''
        import scipy.interpolate as si

        Kind = {2 : 'linear', 3 : 'quadratic'}.get(len(theta), 'cubic')

        return si.interp1d(theta, r, kind = Kind)

    def _SemiCircularArchNodes(self, el_num, r_base, r_design, t, Span = None):
        '''
        Parameters
        ----------
        el_num : Number of elements.
        r_base : Radius at ends.
        r_design : design points radius.
        t : thickness.
        Span : Included angle in degrees, or None for ArchSpan.

        Returns
        -------
        Nodes : Node array for the arch through these design radii.
        DesignX, DesignY : Positions of the design points.

        '''
        NumVar = len(r_design)

        # Design points sit at equal angle increments with the ends excluded,
        # since those are held at r_base.
        theta = self._ArchAngles(NumVar + 2, Span)

        r = np.zeros(2+NumVar)
        r[[0,-1]] = r_base
        r[1:-1] = r_design

        DesignX = r*np.cos(theta)
        DesignY = r*np.sin(theta)

        func = self._RadiusSpline(theta, r)

        app_theta = self._ArchAngles(el_num + 1, Span)
        app_r = func(app_theta)

        NodesX_top = app_r*np.cos(app_theta)
        NodesY_top = app_r*np.sin(app_theta)

        NodesX_bot = (app_r - t)*np.cos(app_theta)
        NodesY_bot = (app_r - t)*np.sin(app_theta)

        Nodes = np.zeros((2*el_num+2,3))
        cnt_top = 0
        cnt_bot = 0
        for i in range(1,2*el_num+3,1):

            if i%2 != 0:

                Nodes[i-1,:] = [i, NodesX_bot[cnt_bot], NodesY_bot[cnt_bot]]
                cnt_bot += 1

            if i%2 == 0:

                Nodes[i-1,:] = [i, NodesX_top[cnt_top], NodesY_top[cnt_top]]
                cnt_top += 1

        return Nodes, DesignX, DesignY


    def SingleElement(self, Length, Height, Load):

        self.Nodes = np.zeros((4,3))
        self.Nodes[:,0] = np.arange(1, 5, 1).astype('int')

        self.Nodes[:,1:] = np.array([[0, 0],
                                [1, 0],
                                [1, 1],
                                [0, 1]])

        self.Nodes[:,1] *= Length
        self.Nodes[:,2] *= Height

        self.Elements = np.array([[1, 1, 2, 3, 4]]).astype('int')

        DegOfFreedom = np.arange(0, self.Nodes[-1,0]*2 , 1).reshape(-1,1)
        self.DegOfFreedom = np.delete(DegOfFreedom, np.array([0, 1, 6])).astype('int')


        self.Load = np.zeros((self.Nodes.shape[0]*2, 1))
        self.Load[2*2 - 1 , 0] = Load
        self.Load[3*2 - 1 , 0] = Load

        self.LoadNode = 3*2 - 1 #y degree of freedom of node 3

        self.VariableNumber = 1

        Nodes_fd = np.zeros((4,3))
        Nodes_fd[:,0] = np.arange(1, 5, 1).astype('int')

        Nodes_fd[:,1:] = np.array([[0, 0],
                                [1, 0],
                                [1, 1],
                                [0, 1]])

        Nodes_fd[:,1] *= Length
        Nodes_fd[:,2] *=  (Height + 1e-6)

        self.dXdx = (Nodes_fd - self.Nodes)[:,[1,2]]
        self.dXdx /= 1e-6

        return

    def SimpleBeam(self, el_num, Length, Height, Load):
        '''
        A simple 1 element thick cantilever beam. Pinned at the left edge and point loaded at the right.

        Parameters
        ----------
        el_num : Number of elements.
        Length : Length of the beam
        Height : Height of the beam.

        Returns
        -------
        None.

        '''
        self.el_num = el_num

        NodesX = np.linspace(0, Length, self.el_num + 1)
        NodesY = np.linspace(0, Height, 2)

        Nodes = np.zeros(((self.el_num + 1)*2, 3))

        Nodes[:,0] = np.arange(1, (self.el_num+1)*2 + 1)

        Nodes[:self.el_num+1, 1] = NodesX
        Nodes[self.el_num + 1:, 1] = NodesX

        Nodes[:self.el_num+1, 2] = NodesY[0]
        Nodes[self.el_num + 1:, 2] = NodesY[1]

        self.Nodes = Nodes

        Elements = np.zeros((self.el_num, 5))

        Elements[:,0] = np.arange(1, self.el_num+1)

        LocalNodes = np.array([1, 2, self.el_num+3, self.el_num+2])

        Elements[0, 1:] = LocalNodes
        for i in np.arange(1, self.el_num):

            LocalNodes += 1

            Elements[i,1:] = LocalNodes

        self.Elements = Elements.astype('int')

        self.Nodal_coor = Nodes[:,[0]]
        self.var_num = 2
        self.LoadNode = Nodes[-1,0]

        #BC
        DegOfFreedom = np.arange(0, Nodes[-1,0]*2 , 1).reshape(-1,1)
        self.DegOfFreedom = np.delete(DegOfFreedom, np.array([0, 1, 2*self.el_num + 2])).astype('int')

        self.LoadNode = Nodes[-1, 0].astype('int')*2 - 1

        self.Load = np.zeros((self.Nodes.shape[0]*2,1))
        self.Load[-1,0] = Load

        # Sensitivity

        fd_Length = Length + 1e-6

        NodesX = np.linspace(0, fd_Length, self.el_num + 1)
        NodesY = np.linspace(0, Height, 2)

        Nodes_fd = np.zeros(((self.el_num + 1)*2, 3))

        Nodes_fd[:,0] = np.arange(1, (self.el_num+1)*2 + 1)

        Nodes_fd[:self.el_num + 1, 1] = NodesX
        Nodes_fd[self.el_num + 1:, 1] = NodesX

        Nodes_fd[:self.el_num + 1, 2] = NodesY[0]
        Nodes_fd[self.el_num + 1:, 2] = NodesY[1]

        self.dXdx = (Nodes_fd - Nodes)/1e-6
        self.dXdx = self.dXdx[:,[1,2]]

        fd_height = Height + 1e-6

        NodesX = np.linspace(0, Length, self.el_num + 1)
        NodesY = np.linspace(0, fd_height, 2)

        Nodes_fd = np.zeros(((self.el_num + 1)*2, 3))

        Nodes_fd[:,0] = np.arange(1, (self.el_num+1)*2 + 1)

        Nodes_fd[:self.el_num + 1, 1] = NodesX
        Nodes_fd[self.el_num + 1:, 1] = NodesX

        Nodes_fd[:self.el_num + 1, 2] = NodesY[0]
        Nodes_fd[self.el_num + 1:, 2] = NodesY[1]

        h_fd = (Nodes_fd - Nodes)/1e-6

        self.dXdx = np.hstack((self.dXdx, h_fd[:,[1,2]]))

        self.VariableNumber =2

        return

    def LeeFrame(self, el_num, var, Load):
        '''
        Parameters
        ----------
        var : array of the two member lengths.

        Returns
        -------
        Two derivative vectors aswell as writes the input file.

        '''
        self.el_num = el_num

        Nodes = self._LeeFrameNodes(self.el_num, var)

        Elements = np.zeros((self.el_num,5))
        Elements[:,0] = np.arange(1,self.el_num+1,1)

        element_up = np.array([1,2,4,3])

        for i in range(0,int(self.el_num/2),1):

            Elements[i,1:] = element_up
            element_up += np.array([2,2,2,2])

        element_side = element_up + np.array([1,4,1,1]) - np.array([2,2,2,2])

        Elements[int(self.el_num/2),1:] = element_side
        element_side += np.array([4,2,2,1])
        Elements[int(self.el_num/2)+1,1:] = element_side

        for i in range(int(self.el_num/2)+2,self.el_num,1):

            element_side += np.array([2,2,2,2])
            Elements[i,1:] = element_side

        self.Nodes = Nodes
        self.Elements = Elements.astype('int')
        self.LoadNode = int(0.66*(Nodes.shape[0]))*2 + 1

        #BC
        DegOfFreedom = np.arange(0, Nodes[-1,0]*2 , 1).reshape(-1,1)
        self.DegOfFreedom = np.delete(DegOfFreedom, np.array([0, 1,
                                                              Nodes[-1,0]*2 - 2,
                                                              Nodes[-1,0]*2 - 1]).astype('int')).astype('int')

        self.Load = np.zeros((self.Nodes.shape[0]*2,1))
        self.Load[self.LoadNode,0] = Load

        self.VariableNumber = 2

        # Sensitivity
        self.dXdx = self._NodeSensitivity(lambda v: self._LeeFrameNodes(self.el_num, v), var)

        return


    def SemiCircularArch(self, el_num, r_base, r_design, t, Load, Span = None):
        '''
        Parameters
        ----------
        el_num : Number of elements along the arch.
        r_base : Radius at ends
        r_design : design points radius
        t : thickness.
        Load : Point load applied at the crown.
        Span : Included angle in degrees, or None for ArchSpan.

        Returns
        -------
        Nodes and Element Matrixes as well as sensitivity.

        '''
        Nodes, DesignX, DesignY = self._SemiCircularArchNodes(el_num, r_base, r_design, t, Span)

        Elements = np.zeros((el_num,5))
        element = np.array([1,1,2,4,3])
        for i in range(1,el_num+1,1):

            Elements[i-1,:] = element

            element += np.array([1,2,2,2,2])


        self.Nodes = Nodes
        self.Elements = Elements.astype('int')
        self.var_num = len(r_design)
        self.LoadNode = int(Nodes[el_num,0])*2 - 1
        self.VarPos = np.vstack((DesignX, DesignY))

        #BC
        DegOfFreedom = np.arange(0, Nodes[-1,0]*2 , 1).reshape(-1,1)
        self.DegOfFreedom = np.delete(DegOfFreedom, np.array([0, 1,
                                                              2, 3,
                                                              Nodes[-1,0]*2 - 2,
                                                              Nodes[-1,0]*2 - 1]).astype('int')).astype('int')

        self.Load = np.zeros((self.Nodes.shape[0]*2,1))
        self.Load[self.LoadNode,0] = Load

        self.VariableNumber = len(r_design)

        # Sensitivity
        self.dXdx = self._NodeSensitivity(
            lambda v: self._SemiCircularArchNodes(el_num, r_base, v, t, Span)[0], r_design)

        return

#Q8 versions of the meshers
class Q8Mesh(Mesh):

    def SingleElement(self, Length, Height, Load):

        self.Nodes = np.zeros((8,3))
        self.Nodes[:,0] = np.arange(1, 9, 1).astype('int')

        self.Nodes[:,1:] = np.array([[0, 0],
                                    [1, 0],
                                    [1, 1],
                                    [0, 1],
                                    [0.5, 0],
                                    [1, 0.5],
                                    [0.5, 1],
                                    [0, 0.5]])

        self.Nodes[:,1] *= Length
        self.Nodes[:,2] *= Height

        self.Elements = np.array([[1, 1, 2, 3, 4, 5, 6, 7, 8]]).astype('int')

        DegOfFreedom = np.arange(0, self.Nodes[-1,0]*2 , 1).reshape(-1,1)
        self.DegOfFreedom = np.delete(DegOfFreedom, np.array([0, 1, 6, 14])).astype('int')

        self.Load = np.zeros((8*2,1))

        self.Load[2*2 - 2 , 0] = 1/6*Load

        self.Load[3*2 - 2 , 0] = 1/6*Load

        self.Load[6*2 - 2 , 0] = 2/3*Load

        self.LoadNode = 3*2 - 2 #x degree of freedom of node 3

        self.var_num = 1
        self.VariableNumber = 1

        Nodes_fd = np.zeros((8,3))
        Nodes_fd[:,0] = np.arange(1, 9, 1).astype('int')

        Nodes_fd[:,1:] = np.array([[0, 0],
                                    [1, 0],
                                    [1, 1],
                                    [0, 1],
                                    [0.5, 0],
                                    [1, 0.5],
                                    [0.5, 1],
                                    [0, 0.5]])

        Nodes_fd[:,1] *= (Length + 1e-6)
        Nodes_fd[:,2] *= Height

        dXdx = (Nodes_fd - self.Nodes)/1e-6

        self.dXdx = dXdx[:,1:]

        return

    def SimpleBeam(self, el_num, Length, Height, Load):
        '''
        A simple 1 element thick cantilever beam. Pinned at the left edge and point loaded at the right.

        Parameters
        ----------
        el_num : Number of elements.
        Length : Length of the beam
        Height : Height of the beam.

        Returns
        -------
        None.

        '''
        self.el_num = el_num
        NodesX = np.linspace(0, Length, el_num*2 + 1)
        NodesY = np.linspace(0, Height, 3)

        Nodes = np.zeros((el_num*5 + 3, 3))

        Nodes[:,0] = np.arange(1, el_num*5 + 4)

        Nodes[:el_num*2 + 1, 1] = NodesX
        Nodes[el_num*2 + 1: el_num*2 + 1 + len(NodesX[0::2]),1] = NodesX[0::2]
        Nodes[el_num*2 + 1 + len(NodesX[0::2]):, 1] = NodesX

        Nodes[:el_num*2 + 1, 2] = NodesY[0]
        Nodes[el_num*2 + 1: el_num*2 + 1 + len(NodesX[0::2]),2] = NodesY[1]
        Nodes[el_num*2 + 1 + len(NodesX[0::2]):, 2] = NodesY[2]

        self.Nodes = Nodes

        GlobalNumber = np.array([1, 3, 3*el_num+5, 3*el_num+3, 2, 2*el_num+3, 3*el_num+4, 2*el_num+2])

        Elements = np.zeros((el_num, 9))
        Elements[:,0] = np.arange(1, el_num+1,1)
        for i in range(el_num):

            Elements[i,1:] = GlobalNumber
            GlobalNumber[0] += 2
            GlobalNumber[1] += 2
            GlobalNumber[4] += 2

            GlobalNumber[5] += 1
            GlobalNumber[7] += 1

            GlobalNumber[2] += 2
            GlobalNumber[3] += 2
            GlobalNumber[6] += 2

        self.Elements = Elements.astype('int')

        self.Nodal_coor = Nodes[:,[0]]
        self.var_num = 2
        self.LoadNode = Nodes[-1,0]

        #BC
        DegOfFreedom = np.arange(0, Nodes[-1,0]*2 , 1).reshape(-1,1)
        self.DegOfFreedom = np.delete(DegOfFreedom, np.array([0, 1, (3*self.el_num + 3)*2 - 2])).astype('int')

        self.LoadNode = Nodes[-1, 0].astype('int')*2 - 1

        self.Load = np.zeros((self.Nodes.shape[0]*2,1))
        self.Load[-1,0] = Load

        # Sensitivity

        fd_Length = Length + 1e-6

        NodesX = np.linspace(0, fd_Length, el_num*2 + 1)
        NodesY = np.linspace(0, Height, 3)

        Nodes_fd = np.zeros((el_num*5 + 3, 3))

        Nodes_fd[:,0] = np.arange(1, el_num*5 + 4)

        Nodes_fd[:el_num*2 + 1, 1] = NodesX
        Nodes_fd[el_num*2 + 1: el_num*2 + 1 + len(NodesX[0::2]),1] = NodesX[0::2]
        Nodes_fd[el_num*2 + 1 + len(NodesX[0::2]):, 1] = NodesX

        Nodes_fd[:el_num*2 + 1, 2] = NodesY[0]
        Nodes_fd[el_num*2 + 1: el_num*2 + 1 + len(NodesX[0::2]),2] = NodesY[1]
        Nodes_fd[el_num*2 + 1 + len(NodesX[0::2]):, 2] = NodesY[2]

        self.dXdx = (Nodes_fd - Nodes)/1e-6
        self.dXdx = self.dXdx[:,[1,2]]

        self.VariableNumber = 1

        return

    def _SemiCircularArchNodes(self, el_num, r_base, r_design, t, Span = None):
        '''
        Parameters
        ----------
        el_num : Number of elements along the arch.
        r_base : Radius at ends.
        r_design : design points radius.
        t : thickness.
        Span : Included angle in degrees, or None for ArchSpan.

        Returns
        -------
        Nodes : Node array for the eight node arch through these design radii.
        DesignX, DesignY : Positions of the design points.

        Notes
        -----
        Same three row layout as Q8Mesh.SimpleBeam, so the connectivity carries
        over unchanged: the inner and outer rows carry a node at every one of
        the 2*el_num + 1 angular stations, and the middle row only at the
        el_num + 1 element boundaries, giving 5*el_num + 3 nodes.

        '''
        NumVar = len(r_design)

        Design = self._ArchAngles(NumVar + 2, Span)

        r = np.zeros(2 + NumVar)
        r[[0,-1]] = r_base
        r[1:-1] = r_design

        DesignX = r*np.cos(Design)
        DesignY = r*np.sin(Design)

        Radius = self._RadiusSpline(Design, r)

        Station = self._ArchAngles(2*el_num + 1, Span)   # corners and mid sides
        Boundary = Station[0::2]                          # element boundaries only

        Outer = Radius(Station)
        Inner = Outer - t
        Middle = Radius(Boundary) - t/2

        Nodes = np.zeros((el_num*5 + 3, 3))
        Nodes[:,0] = np.arange(1, el_num*5 + 4)

        Split = 2*el_num + 1

        # The outer radius goes in the first row. SimpleBeam's connectivity runs
        # first row -> second row, and on an arch that traversal is only counter
        # clockwise, giving a positive Jacobian, if the first row is the outer
        # one. With the rows the other way round every detJ comes out negative
        # and the stiffness matrix changes sign.
        Nodes[:Split, 1] = Outer*np.cos(Station)
        Nodes[:Split, 2] = Outer*np.sin(Station)

        Nodes[Split : Split + el_num + 1, 1] = Middle*np.cos(Boundary)
        Nodes[Split : Split + el_num + 1, 2] = Middle*np.sin(Boundary)

        Nodes[Split + el_num + 1 :, 1] = Inner*np.cos(Station)
        Nodes[Split + el_num + 1 :, 2] = Inner*np.sin(Station)

        return Nodes, DesignX, DesignY

    def SemiCircularArch(self, el_num, r_base, r_design, t, Load, Span = None):
        '''
        Parameters
        ----------
        el_num : Number of elements along the arch.
        r_base : Radius at ends.
        r_design : design points radius.
        t : thickness of the arch in the radial direction.
        Load : Point load applied at the crown.
        Span : Included angle in degrees, or None for ArchSpan.

        Returns
        -------
        None.

        Notes
        -----
        The arch is clamped at the first end, where all three nodes of the end
        edge are held, and pinned at the other, where a single node is held so
        the edge can still rotate. The four node arch pins its outer node
        because a two node edge has no mid side node to pin, and this follows
        that choice so the two element types stay comparable.

        '''
        self.el_num = el_num

        Nodes, DesignX, DesignY = self._SemiCircularArchNodes(el_num, r_base, r_design, t, Span)

        self.Nodes = Nodes

        # Same connectivity generation as Q8Mesh.SimpleBeam: the node layout is
        # identical, only the co-ordinates differ.
        GlobalNumber = np.array([1, 3, 3*el_num+5, 3*el_num+3, 2, 2*el_num+3, 3*el_num+4, 2*el_num+2])

        Elements = np.zeros((el_num, 9))
        Elements[:,0] = np.arange(1, el_num+1, 1)
        for i in range(el_num):

            Elements[i,1:] = GlobalNumber
            GlobalNumber[0] += 2
            GlobalNumber[1] += 2
            GlobalNumber[4] += 2

            GlobalNumber[5] += 1
            GlobalNumber[7] += 1

            GlobalNumber[2] += 2
            GlobalNumber[3] += 2
            GlobalNumber[6] += 2

        self.Elements = Elements.astype('int')

        self.var_num = len(r_design)
        self.VariableNumber = len(r_design)
        self.VarPos = np.vstack((DesignX, DesignY))

        Split = 2*el_num + 1

        #BC: clamped end holds the whole edge, pinned end holds one node.
        # The pin must be the outer node, matching the four node arch: pinning
        # the inner node instead moves the crown deflection by 8 percent, since
        # it shifts the centre the end edge rotates about.
        Clamped = [0, Split, Split + el_num + 1]
        Pinned = [2*el_num]

        Held = []
        for Node in Clamped + Pinned:

            Held += [2*Node, 2*Node + 1]

        DegOfFreedom = np.arange(0, Nodes.shape[0]*2, 1)
        self.DegOfFreedom = np.delete(DegOfFreedom, np.array(Held)).astype('int')

        # Load on the outer node at the crown. Which surface it acts on shifts
        # the answer by under a fifth of a percent.
        self.LoadNode = 2*el_num + 1

        self.Load = np.zeros((Nodes.shape[0]*2, 1))
        self.Load[self.LoadNode, 0] = Load

        # Sensitivity
        self.dXdx = self._NodeSensitivity(
            lambda v: self._SemiCircularArchNodes(el_num, r_base, v, t, Span)[0], r_design)

        return
