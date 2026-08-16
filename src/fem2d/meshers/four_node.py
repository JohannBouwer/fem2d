"""Four node meshers.

Mesh is the catalogue of shipped geometries for the four node element family,
Q4 and the five beta assumed stress element. Each mesher is a node array, a
connectivity array, boundary conditions by node, a load, and a sensitivity
rule; everything else comes from MeshBase.
"""

import numpy as np

from fem2d.meshers.base import MeshBase


class Mesh(MeshBase):

    def Build(self, Mesher = 'SimpleBeam', *Args, **Kwargs) -> None:
        '''
        Parameters
        ----------
        Mesher : Name of the mesher to run, for example 'LeeFrame'.
        *Args, **Kwargs : Passed straight to it.

        Returns
        -------
        None. A convenience for code that picks a geometry by name:
        Build('LeeFrame', 10, [10.0, 10.0], -25.0).

        '''

        getattr(self, Mesher)(*Args, **Kwargs)

        return

    # ------------------------------------------------------------------
    # Node layouts
    # ------------------------------------------------------------------

    def _SingleElementNodes(self, Length, Height):
        '''
        Parameters
        ----------
        Length, Height : Side lengths of the element.

        Returns
        -------
        Nodes : Node array for the unit square scaled to this size.

        '''

        return self._GridNodes([([0, Length], [0, 0]),
                                ([Length, 0], [Height, Height])])

    def _SimpleBeamNodes(self, el_num, Length, Height):
        '''
        Parameters
        ----------
        el_num : Number of elements along the beam.
        Length, Height : Size of the beam.

        Returns
        -------
        Nodes : Two rows of el_num + 1 nodes, bottom row first.

        '''
        NodesX = np.linspace(0, Length, el_num + 1)

        return self._GridNodes([(NodesX, 0.0),
                                (NodesX, Height)])

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
        t = self.LeeFrameDepth
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

    def _ArchDesign(self, r_base, r_design, Span = None):
        '''
        Parameters
        ----------
        r_base : Radius at the two ends.
        r_design : Radii at the design points.
        Span : Included angle in degrees, or None for ArchSpan.

        Returns
        -------
        Radius : Interpolant giving the radius at any angle.
        DesignX, DesignY : Positions of the design points.

        Notes
        -----
        Design points sit at equal angle increments with the ends excluded,
        since those are held at r_base.

        '''
        NumVar = len(r_design)

        theta = self._ArchAngles(NumVar + 2, Span)

        r = np.zeros(2 + NumVar)
        r[[0,-1]] = r_base
        r[1:-1] = r_design

        return self._RadiusSpline(theta, r), r*np.cos(theta), r*np.sin(theta)

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
        Nodes : Node array for the arch through these design radii, the inner
                and outer surfaces interleaved so the connectivity runs round
                the arch two nodes at a time.
        DesignX, DesignY : Positions of the design points.

        '''
        Radius, DesignX, DesignY = self._ArchDesign(r_base, r_design, Span)

        Station = self._ArchAngles(el_num + 1, Span)

        Outer = Radius(Station)
        Inner = Outer - t

        Nodes = self._GridNodes([(Inner*np.cos(Station), Inner*np.sin(Station)),
                                 (Outer*np.cos(Station), Outer*np.sin(Station))])

        # Interleave the two surfaces: inner, outer, inner, outer round the arch.
        Interleaved = np.empty_like(Nodes)
        Interleaved[0::2, 1:] = Nodes[:el_num + 1, 1:]
        Interleaved[1::2, 1:] = Nodes[el_num + 1:, 1:]
        Interleaved[:, 0] = Nodes[:, 0]

        return Interleaved, DesignX, DesignY

    # ------------------------------------------------------------------
    # Meshers
    # ------------------------------------------------------------------

    def SingleElement(self, Length, Height, Load):
        '''
        Parameters
        ----------
        Length, Height : Side lengths of the element.
        Load : Applied at both top nodes.

        Returns
        -------
        None.

        '''
        self.SetNodes(self._SingleElementNodes(Length, Height))
        self.SetElements(self._StripElements([1, 2, 3, 4], [0, 0, 0, 0], 1))

        self.Pin(1)
        self.Fix(4, 'x')

        self.AddLoad([2, 3], [0.0, Load])

        # The load is shared between the two top nodes, so name the one the
        # load path follows rather than letting the default pick either.
        self.SetProbe(Node = 3, Direction = 'y')

        self.SetSensitivity(lambda v: self._SingleElementNodes(Length, v[0]), [Height])

        self._Finalise()

        return

    def SimpleBeam(self, el_num, Length, Height, Load):
        '''
        A simple 1 element thick cantilever beam. Pinned at the left edge and point loaded at the right.

        Parameters
        ----------
        el_num : Number of elements.
        Length : Length of the beam
        Height : Height of the beam.
        Load : Point load at the free end.

        Returns
        -------
        None.

        '''
        self.el_num = el_num

        self.SetNodes(self._SimpleBeamNodes(el_num, Length, Height))
        self.SetElements(self._StripElements([1, 2, el_num + 3, el_num + 2], [1, 1, 1, 1], el_num))

        self.Pin(1)
        self.Fix(el_num + 2, 'x')

        self.AddLoad(self.Nodes.shape[0], [0.0, Load])

        self.SetSensitivity(lambda v: self._SimpleBeamNodes(el_num, v[0], v[1]), [Length, Height])

        self._Finalise()

        return

    def LeeFrame(self, el_num, var, Load):
        '''
        Parameters
        ----------
        el_num : Number of elements.
        var : array of the two member lengths.
        Load : Point load on the horizontal member.

        Returns
        -------
        None.

        '''
        self.el_num = el_num

        self.SetNodes(self._LeeFrameNodes(el_num, var))

        Elements = np.zeros((el_num, 4), dtype = 'int')

        Elements[:int(el_num/2)] = self._StripElements([1, 2, 4, 3], [2, 2, 2, 2],
                                                       int(el_num/2))

        # The corner turns from the vertical member into the horizontal one, so
        # its two elements do not follow the strip pattern either side of it.
        element_side = np.array([el_num, el_num + 4, el_num + 3, el_num + 2])
        Elements[int(el_num/2)] = element_side

        element_side = element_side + np.array([4, 2, 2, 1])
        Elements[int(el_num/2) + 1] = element_side

        Elements[int(el_num/2) + 2:] = self._StripElements(element_side + np.array([2, 2, 2, 2]),
                                                           [2, 2, 2, 2],
                                                           el_num - int(el_num/2) - 2)

        self.SetElements(Elements)

        self.Pin([1, self.Nodes.shape[0]])

        # The classic Lee frame load sits about two thirds of the way along the
        # horizontal member from the corner.
        LoadNode = int(0.66*self.Nodes.shape[0]) + 1
        self.AddLoad(LoadNode, [0.0, Load])

        self.SetSensitivity(lambda v: self._LeeFrameNodes(el_num, v), var)

        self._Finalise()

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
        None.

        '''
        self.el_num = el_num

        Nodes, DesignX, DesignY = self._SemiCircularArchNodes(el_num, r_base, r_design, t, Span)

        self.SetNodes(Nodes)
        self.SetElements(self._StripElements([1, 2, 4, 3], [2, 2, 2, 2], el_num))

        self.SetDesignPoints(DesignX, DesignY)

        # Clamped at the first end, where both nodes of the end edge are held,
        # and pinned at the other, where a single node is held so the edge can
        # still rotate.
        self.Clamp([1, 2])
        self.Pin(self.Nodes.shape[0])

        self.AddLoad(el_num + 1, [0.0, Load])

        self.SetSensitivity(
            lambda v: self._SemiCircularArchNodes(el_num, r_base, v, t, Span)[0], r_design)

        self._Finalise()

        return
