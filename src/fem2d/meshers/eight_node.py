"""Eight node meshers.

Q8Mesh is the same catalogue of geometries as Mesh, laid out for the eight node
element. Every geometry here uses the same three row node layout: the two
surfaces carry a node at each of the 2*el_num + 1 corner and mid side stations,
and the row between them only at the el_num + 1 element boundaries, giving
5*el_num + 3 nodes and one connectivity pattern shared by all of them.
"""

import numpy as np

from fem2d.meshers.four_node import Mesh


class Q8Mesh(Mesh):
    def _Q8Strip(self, el_num):
        """
        Parameters
        ----------
        el_num : Number of elements in the strip.

        Returns
        -------
        Connectivity for a strip of eight node elements over the three row node
        layout. The two surface rows step two nodes per element, the middle row
        one, which is the whole difference between the columns.

        """

        return self._StripElements(
            [1, 3, 3 * el_num + 5, 3 * el_num + 3, 2, 2 * el_num + 3, 3 * el_num + 4, 2 * el_num + 2],
            [2, 2, 2, 2, 2, 1, 2, 1],
            el_num,
        )

    def _SingleElementNodes(self, Length, Height):
        """
        Parameters
        ----------
        Length, Height : Side lengths of the element.

        Returns
        -------
        Nodes : Four corners then four mid side nodes, in local node order.

        """

        return self._GridNodes(
            [
                ([0, Length, Length, 0], [0, 0, Height, Height]),
                ([0.5 * Length, Length, 0.5 * Length, 0], [0, 0.5 * Height, Height, 0.5 * Height]),
            ]
        )

    def _SimpleBeamNodes(self, el_num, Length, Height):
        """
        Parameters
        ----------
        el_num : Number of elements along the beam.
        Length, Height : Size of the beam.

        Returns
        -------
        Nodes : Three rows, the bottom surface first.

        """
        NodesX = np.linspace(0, Length, el_num * 2 + 1)

        Bottom = np.column_stack((NodesX, np.zeros_like(NodesX)))
        Top = np.column_stack((NodesX, np.full_like(NodesX, Height)))
        Middle = np.column_stack((NodesX[0::2], np.full_like(NodesX[0::2], 0.5 * Height)))

        return self._StripNodes(Bottom, Top, Middle=Middle)

    def _SemiCircularArchNodes(self, el_num, r_base, r_design, t, Span=None):
        """
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
        The outer radius goes in the first row. The connectivity runs first row
        to second row, and on an arch that traversal is only counter clockwise,
        giving a positive Jacobian, if the first row is the outer one. With the
        rows the other way round every detJ comes out negative and the
        stiffness matrix changes sign.

        """
        Radius, DesignX, DesignY = self._ArchDesign(r_base, r_design, Span)

        Station = self._ArchAngles(2 * el_num + 1, Span)  # corners and mid sides
        Boundary = Station[0::2]  # element boundaries only

        Outer = Radius(Station)
        Inner = Outer - t
        Middle = Radius(Boundary) - t / 2

        Nodes = self._StripNodes(
            np.column_stack((Outer * np.cos(Station), Outer * np.sin(Station))),
            np.column_stack((Inner * np.cos(Station), Inner * np.sin(Station))),
            Middle=np.column_stack((Middle * np.cos(Boundary), Middle * np.sin(Boundary))),
        )

        return Nodes, DesignX, DesignY

    def _LeeFrameSurfaces(self, el_num, var):
        """
        Parameters
        ----------
        el_num : Number of elements around the frame.
        var : The two member lengths, vertical then horizontal.

        Returns
        -------
        Inner, Outer : (2*el_num + 1, 2) co-ordinates of the two surfaces of
                       the L shaped frame, at every corner and mid side
                       station. Inner comes first because that is the order
                       that makes the strip traverse counter clockwise; the
                       other way round every element comes out with a negative
                       area, which Validate rejects.

        Notes
        -----
        The elements are split between the two members the same way the four
        node frame splits them, so the two element types put the corner in the
        same place. The corner itself is mitred: the inner surface turns at
        (t, length_up - t) while the outer turns at (0, length_up), so the
        element edge between them runs diagonally across the corner.

        """
        length_up, length_side = var[0], var[1]
        t = self.LeeFrameDepth

        # The four node frame gives the vertical member int((el_num-1)/2)+1
        # elements and the horizontal one the rest.
        Up = int((el_num - 1) / 2) + 1
        Side = el_num - Up

        UpStations = np.linspace(0, length_up, 2 * Up + 1)
        SideStations = np.linspace(0, length_side, 2 * Side + 1)

        # Outer surface: up the left face of the vertical member, then along
        # the top face of the horizontal one.
        Outer = np.vstack(
            (
                np.column_stack((np.zeros_like(UpStations), UpStations)),
                np.column_stack((SideStations[1:], np.full_like(SideStations[1:], length_up))),
            )
        )

        # Inner surface: up the right face, turning the corner a depth early.
        Inner = np.vstack(
            (
                np.column_stack((np.full_like(UpStations, t), UpStations * (length_up - t) / length_up)),
                np.column_stack(
                    (
                        t + SideStations[1:] * (length_side - t) / length_side,
                        np.full_like(SideStations[1:], length_up - t),
                    )
                ),
            )
        )

        return Inner, Outer

    def SingleElement(self, Length, Height, Load):
        """
        Parameters
        ----------
        Length, Height : Side lengths of the element.
        Load : Total load, spread over the loaded edge by the consistent nodal
               values for a quadratic edge, one sixth at each corner and two
               thirds at the mid side node.

        Returns
        -------
        None.

        """
        self.SetNodes(self._SingleElementNodes(Length, Height))
        self.SetElements(self._StripElements([1, 2, 3, 4, 5, 6, 7, 8], np.zeros(8, dtype="int"), 1))

        self.Pin(1)
        self.Fix([4, 8], "x")

        self.AddLoad([2, 3], [1 / 6 * Load, 0.0])
        self.AddLoad(6, [2 / 3 * Load, 0.0])

        # The load is spread over three nodes, so name the one the load path
        # follows rather than letting the default pick the largest share.
        self.SetProbe(Node=3, Direction="x")

        self.SetSensitivity(lambda v: self._SingleElementNodes(v[0], Height), [Length])

        self._Finalise()

        return

    def SimpleBeam(self, el_num, Length, Height, Load):
        """
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

        """
        self.el_num = el_num

        self.SetNodes(self._SimpleBeamNodes(el_num, Length, Height))
        self.SetElements(self._Q8Strip(el_num))

        self.Pin(1)
        self.Fix(3 * el_num + 3, "x")

        self.AddLoad(self.Nodes.shape[0], [0.0, Load])

        self.SetSensitivity(lambda v: self._SimpleBeamNodes(el_num, v[0], v[1]), [Length, Height])

        self._Finalise()

        return

    def LeeFrame(self, el_num, var, Load):
        """
        Parameters
        ----------
        el_num : Number of elements.
        var : array of the two member lengths.
        Load : Point load on the horizontal member.

        Returns
        -------
        None.

        """
        self.el_num = el_num

        self.SetNodes(self._StripNodes(*self._LeeFrameSurfaces(el_num, var)))
        self.SetElements(self._Q8Strip(el_num))

        # One node held at each end, at the same two physical corners the four
        # node frame pins: the outer node of the base edge, id 3*el_num + 3,
        # and the inner node of the free end edge, id 2*el_num + 1.
        self.Pin([3 * el_num + 3, 2 * el_num + 1])

        self.AddLoad(self._LeeFrameLoadNode(el_num, var), [0.0, Load])

        self.SetSensitivity(lambda v: self._StripNodes(*self._LeeFrameSurfaces(el_num, v)), var)

        self._Finalise()

        return

    def _LeeFrameLoadNode(self, el_num, var):
        """
        Parameters
        ----------
        el_num : Number of elements.
        var : The two member lengths.

        Returns
        -------
        The one based id of the outer surface node closest to where the four
        node frame puts its load, which is on the top face of the horizontal
        member. Matching the physical position rather than the index
        arithmetic is what keeps the two element types comparable, the same
        reasoning the arch uses for its pinned end.

        """
        Q4Nodes = self._LeeFrameNodes(el_num, var)
        Target = Q4Nodes[int(0.66 * Q4Nodes.shape[0]), 1:]

        _, Outer = self._LeeFrameSurfaces(el_num, var)

        # The outer surface is the third row of the strip, so its node ids
        # start after the inner surface and the middle row.
        return 3 * el_num + 3 + int(np.argmin(np.sum((Outer - Target) ** 2, axis=1)))

    def SemiCircularArch(self, el_num, r_base, r_design, t, Load, Span=None):
        """
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

        """
        self.el_num = el_num

        Nodes, DesignX, DesignY = self._SemiCircularArchNodes(el_num, r_base, r_design, t, Span)

        self.SetNodes(Nodes)
        self.SetElements(self._Q8Strip(el_num))

        self.SetDesignPoints(DesignX, DesignY)

        Split = 2 * el_num + 1

        self.Clamp([1, Split + 1, Split + el_num + 2])
        self.Pin(2 * el_num + 1)

        # Load on the outer node at the crown. Which surface it acts on shifts
        # the answer by under a fifth of a percent.
        self.AddLoad(el_num + 1, [0.0, Load])

        self.SetSensitivity(lambda v: self._SemiCircularArchNodes(el_num, r_base, v, t, Span)[0], r_design)

        self._Finalise()

        return
