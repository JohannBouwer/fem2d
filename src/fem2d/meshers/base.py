"""Abstract base class for every mesh in the library.

A concrete mesh supplies its geometry only: an array of nodes, an array of
elements, which degrees of freedom are held, and where the load acts.
Everything else, from the free degree of freedom bookkeeping through to the
shape sensitivities and the validity checks, is derived here and shared by
every mesh.

To add a mesh, subclass MeshBase and implement Build. A mesh is four calls:

    class MyMesh(MeshBase):

        def Build(self, el_num, Length, Height, Load):

            self.SetNodes(self._GridNodes([(np.linspace(0, Length, el_num + 1), 0.0),
                                           (np.linspace(0, Length, el_num + 1), Height)]))
            self.SetElements(self._StripElements([1, 2, el_num + 3, el_num + 2],
                                                 [1, 1, 1, 1], el_num))
            self.Clamp([1, el_num + 2])
            self.AddLoad(2*el_num + 2, [0.0, Load])
            self.SetSensitivity(lambda v: ..., [Length, Height])

            self._Finalise()

            return

See fem2d/meshers/four_node.py for the shipped examples and
notebooks/Example_custom_mesh.ipynb for a worked one.
"""

import warnings
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from fem2d.elements.registry import LookupElement
from fem2d.materials import PlaneState


class MeshWarning(UserWarning):
    '''Issued by Validate for a mesh that is buildable but probably wrong.'''


# node*2 - 2 : x direction
# node*2 - 1 : y direction
class MeshBase(ABC):

    #: Forward difference step used by every nodal sensitivity.
    FDStep: ClassVar[float] = 1e-6

    #: Run Validate from _Finalise. Set False on the class or on an instance to
    #: build a mesh the checks reject.
    AutoValidate: ClassVar[bool] = True

    #: Included angle of the arch in degrees. 215 is the DaDeppo and Schmidt
    #: deep circular arch, which is the geometry the shape design work uses.
    ArchSpan: ClassVar[float] = 215.0

    #: Depth of the Lee frame members.
    LeeFrameDepth: ClassVar[float] = 2.0

    #: Relative tolerance below which two nodes count as sitting on top of each
    #: other, and an element counts as degenerate.
    Tolerance: ClassVar[float] = 1e-12

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
                    option: a registered name such as 'Q4', 'Q8' or '5B', or an
                    Element subclass directly.

        Returns
        -------
        None.

        '''
        self.E = E
        self.v = v
        self.t = thickness
        self.plane = PlaneState.From(plane)

        self.ElementType = ElementType

        # Degrees of freedom held by a boundary condition. Kept as a set so a
        # node can be held twice, by two overlapping calls, without counting
        # twice.
        self._Held: set[int] = set()

        # Degree of freedom the load path plots track. Defaulted in _Finalise
        # from the applied load when a mesher does not choose one.
        self.ProbeDOF: int | None = None

        return

    @abstractmethod
    def Build(self, *Args, **Kwargs) -> None:
        '''
        Build the mesh: set the nodes and elements, apply the boundary
        conditions and the load, and call _Finalise.

        Returns
        -------
        None. Everything is written onto self.
        '''

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def SetNodes(self, Nodes) -> None:
        '''
        Parameters
        ----------
        Nodes : (NumNodes, 3) array of [id, x, y], ids running 1 to NumNodes.

        Returns
        -------
        None. Stores the nodes and sizes everything that depends on the node
        count: the load vector, the free degrees of freedom, and the set of
        held degrees of freedom, which is cleared. Call this before applying
        any boundary condition or load.

        '''
        self.Nodes = np.asarray(Nodes, dtype = float)

        self._Held = set()
        self.Load = np.zeros((self.Nodes.shape[0]*2, 1))

        self._RebuildDegOfFreedom()

        return

    def SetElements(self, Connectivity) -> None:
        '''
        Parameters
        ----------
        Connectivity : (NumElements, NumNodes) array of one based node ids, or
                       (NumElements, 1 + NumNodes) with the element id column
                       already in place.

        Returns
        -------
        None.

        '''
        Connectivity = np.asarray(Connectivity)

        if Connectivity.shape[1] == self.NumElementNodes:

            Ids = np.arange(1, Connectivity.shape[0] + 1).reshape(-1, 1)
            Connectivity = np.hstack((Ids, Connectivity))

        self.Elements = Connectivity.astype('int')

        return

    @property
    def NumElementNodes(self) -> int:
        '''
        Returns
        -------
        Number of nodes the element type carries, which fixes how wide the
        connectivity has to be.
        '''

        return LookupElement(self.ElementType).NumNodes

    def _GridNodes(self, Rows) -> npt.NDArray[np.float64]:
        '''
        Parameters
        ----------
        Rows : Sequence of (X, Y) pairs, one per row of nodes, stacked in the
               order given. Either member of a pair may be a scalar, which is
               then held constant along that row.

        Returns
        -------
        Nodes : (NumNodes, 3) array of [id, x, y] with the ids numbered from 1.

        '''
        Coordinates = []

        for X, Y in Rows:

            X, Y = np.broadcast_arrays(np.asarray(X, dtype = float),
                                       np.asarray(Y, dtype = float))

            Coordinates.append(np.column_stack((np.ravel(X), np.ravel(Y))))

        Coordinates = np.vstack(Coordinates)

        Nodes = np.zeros((Coordinates.shape[0], 3))
        Nodes[:, 0] = np.arange(1, Coordinates.shape[0] + 1)
        Nodes[:, 1:] = Coordinates

        return Nodes

    def _StripNodes(self, First, Second, Middle = None) -> npt.NDArray[np.float64]:
        '''
        Parameters
        ----------
        First : (2*el_num + 1, 2) co-ordinates along one surface, at every
                corner and mid side station.
        Second : (2*el_num + 1, 2) co-ordinates along the opposite surface.
        Middle : (el_num + 1, 2) co-ordinates of the row between them, which
                 carries a node only at the element boundaries. Defaults to the
                 midpoint of the two surfaces there.

        Returns
        -------
        Nodes : (5*el_num + 3, 3) three row node array for an eight node strip,
                laid out first surface, middle row, second surface.

        Notes
        -----
        First must be the surface that makes the traversal counter clockwise.
        With the two surfaces the other way round every detJ comes out negative
        and the stiffness matrix changes sign, which Validate catches as a
        negative element area.

        '''
        First = np.asarray(First, dtype = float)
        Second = np.asarray(Second, dtype = float)

        if Middle is None:

            Middle = 0.5*(First[0::2] + Second[0::2])

        Middle = np.asarray(Middle, dtype = float)

        return self._GridNodes([(First[:, 0], First[:, 1]),
                                (Middle[:, 0], Middle[:, 1]),
                                (Second[:, 0], Second[:, 1])])

    def _MidSideStations(self, Corners) -> npt.NDArray[np.float64]:
        '''
        Parameters
        ----------
        Corners : (el_num + 1, 2) co-ordinates of the element boundaries along
                  one surface.

        Returns
        -------
        (2*el_num + 1, 2) with the midpoint of each pair inserted between them,
        which is the station array _StripNodes wants. Putting the mid side node
        at the midpoint of the edge it belongs to keeps the edge straight, so
        the element stays well shaped even where consecutive boundaries are
        unevenly spaced.

        '''
        Corners = np.asarray(Corners, dtype = float)

        Stations = np.zeros((2*Corners.shape[0] - 1, 2))
        Stations[0::2] = Corners
        Stations[1::2] = 0.5*(Corners[:-1] + Corners[1:])

        return Stations

    def _StripElements(self, First, Increment, Count) -> npt.NDArray[np.int64]:
        '''
        Parameters
        ----------
        First : Node ids of the first element, in local node order.
        Increment : Added to those ids to step from one element to the next.
        Count : Number of elements in the strip.

        Returns
        -------
        Elements : (Count, NumNodes) connectivity, ready for SetElements, which
                   adds the id column.

        '''
        First = np.asarray(First, dtype = 'int')
        Increment = np.asarray(Increment, dtype = 'int')

        Steps = np.arange(Count).reshape(-1, 1)

        return First + Steps*Increment

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _DirectionDOF(self, Nodes, Direction) -> npt.NDArray[np.int64]:
        '''
        Parameters
        ----------
        Nodes : One based node id, or an iterable of them.
        Direction : 'x', 'y' or 'xy'.

        Returns
        -------
        The global degrees of freedom those nodes span in that direction.

        '''
        Nodes = np.atleast_1d(np.asarray(Nodes, dtype = 'int'))

        Direction = str(Direction).lower()

        if Direction not in ('x', 'y', 'xy', 'yx'):

            raise ValueError(f"Unknown Direction {Direction!r}, expected 'x', 'y' or 'xy'.")

        DOF = []

        if 'x' in Direction:

            DOF.append(Nodes*2 - 2)

        if 'y' in Direction:

            DOF.append(Nodes*2 - 1)

        return np.sort(np.concatenate(DOF))

    def Fix(self, Nodes, Direction = 'xy') -> None:
        '''
        Parameters
        ----------
        Nodes : One based node id, or an iterable of them.
        Direction : 'x', 'y' or 'xy'. The default is both.

        Returns
        -------
        None. Holds those degrees of freedom and rebuilds DegOfFreedom, so the
        free set is never stale.

        '''
        self._Held.update(int(DOF) for DOF in self._DirectionDOF(Nodes, Direction))

        self._RebuildDegOfFreedom()

        return

    def Pin(self, Nodes) -> None:
        '''
        Parameters
        ----------
        Nodes : One based node id, or an iterable of them.

        Returns
        -------
        None. Holds both directions at a single node, so the edge through it
        can still rotate. Mechanically the same as Clamp, named apart because
        the distinction is what the mesher means, not what it does.

        '''

        self.Fix(Nodes, 'xy')

        return

    def Clamp(self, Nodes) -> None:
        '''
        Parameters
        ----------
        Nodes : Every node along the edge being clamped.

        Returns
        -------
        None. Holds both directions at all of them, so the edge cannot rotate.

        '''

        self.Fix(Nodes, 'xy')

        return

    def Free(self, Nodes, Direction = 'xy') -> None:
        '''
        Parameters
        ----------
        Nodes : One based node id, or an iterable of them.
        Direction : 'x', 'y' or 'xy'. The default is both.

        Returns
        -------
        None. Releases those degrees of freedom again.

        '''
        self._Held.difference_update(int(DOF) for DOF in self._DirectionDOF(Nodes, Direction))

        self._RebuildDegOfFreedom()

        return

    def _RebuildDegOfFreedom(self) -> None:
        '''
        Returns
        -------
        None. Rewrites DegOfFreedom as every degree of freedom that is not held.
        '''

        self.DegOfFreedom = np.delete(np.arange(0, self.Nodes.shape[0]*2),
                                      np.array(sorted(self._Held), dtype = 'int')).astype('int')

        return

    @property
    def HeldDOF(self) -> npt.NDArray[np.int64]:
        '''
        Returns
        -------
        Sorted array of the degrees of freedom held by a boundary condition.
        '''

        return np.array(sorted(self._Held), dtype = 'int')

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------

    def AddLoadDOF(self, DOF, Values) -> None:
        '''
        Parameters
        ----------
        DOF : Global degree of freedom index, or an iterable of them.
        Values : Value to add at each, scalar or one per degree of freedom.

        Returns
        -------
        None. Accumulates into Load, so repeated calls add up.

        '''
        DOF = np.atleast_1d(np.asarray(DOF, dtype = 'int'))
        Values = np.broadcast_to(np.asarray(Values, dtype = float), DOF.shape)

        np.add.at(self.Load[:, 0], DOF, Values)

        return

    def AddLoad(self, Nodes, Vector) -> None:
        '''
        Parameters
        ----------
        Nodes : One based node id, or an iterable of them.
        Vector : (2,) [Fx, Fy] applied at every node given, or (NumNodes, 2)
                 with one row per node for a distribution that varies along the
                 set.

        Returns
        -------
        None. Accumulates into Load, so a load can be built up over as many
        calls, nodes and directions as the geometry needs.

        '''
        Nodes = np.atleast_1d(np.asarray(Nodes, dtype = 'int'))
        Vector = np.atleast_2d(np.asarray(Vector, dtype = float))

        if Vector.shape[1] != 2:

            raise ValueError(f'Vector must have two columns, [Fx, Fy], got shape {Vector.shape}.')

        if Vector.shape[0] not in (1, Nodes.size):

            raise ValueError(f'Vector has {Vector.shape[0]} rows, expected 1 or {Nodes.size}, '
                             'one per node.')

        Vector = np.broadcast_to(Vector, (Nodes.size, 2))

        self.AddLoadDOF(Nodes*2 - 2, Vector[:, 0])
        self.AddLoadDOF(Nodes*2 - 1, Vector[:, 1])

        return

    def SetLoad(self, Nodes, Vector) -> None:
        '''
        Parameters
        ----------
        Nodes : One based node id, or an iterable of them.
        Vector : (2,) or (NumNodes, 2), as for AddLoad.

        Returns
        -------
        None. Clears any load already applied, then applies this one.

        '''
        self.Load[:] = 0.0

        self.AddLoad(Nodes, Vector)

        return

    @property
    def LoadDOF(self) -> npt.NDArray[np.int64]:
        '''
        Returns
        -------
        The degrees of freedom carrying a non zero load.
        '''

        return np.flatnonzero(self.Load[:, 0])

    def SetProbe(self, DOF = None, Node = None, Direction = 'y') -> None:
        '''
        Parameters
        ----------
        DOF : Global degree of freedom index to track, if known directly.
        Node : One based node id to track instead.
        Direction : 'x' or 'y' at that node. The default is 'y'.

        Returns
        -------
        None. Sets ProbeDOF, the degree of freedom Plotting.LoadPath and the
        load displacement curves follow. A mesher only needs to call this when
        the default, the largest applied load component, is not the point of
        interest.

        '''
        if (DOF is None) == (Node is None):

            raise ValueError('Give exactly one of DOF or Node.')

        if Node is not None:

            DOF = int(self._DirectionDOF(Node, Direction)[0])

        self.ProbeDOF = int(DOF)

        return

    @property
    def LoadNode(self) -> int:
        '''
        Returns
        -------
        ProbeDOF, under the name it had before loads could span more than one
        node. It is a global degree of freedom index, not a node number,
        despite the name. Kept because postprocessing.Plotting.LoadPath and the
        research notebooks index the displacement history with it.
        '''

        return self.ProbeDOF

    @LoadNode.setter
    def LoadNode(self, DOF) -> None:

        self.ProbeDOF = int(DOF)

        return

    # ------------------------------------------------------------------
    # Sensitivity
    # ------------------------------------------------------------------

    def _NodeSensitivity(self, Rebuild, Variables, step = None):
        '''
        Parameters
        ----------
        Rebuild : Callable taking a list of design variables and returning the
                  Nodes array the mesher would build from them.
        Variables : Current values of the design variables.
        step : Forward difference step on each variable. The default is FDStep.

        Returns
        -------
        dXdx : (NumNodes, 2*NumVariables) derivative of every nodal co-ordinate
               w.r.t each design variable, laid out as the solvers expect.

        '''
        if step is None:

            step = self.FDStep

        Base = Rebuild(list(Variables))

        dXdx = np.zeros((Base.shape[0], 2*len(Variables)))

        for i in range(len(Variables)):

            Perturbed = list(Variables)
            Perturbed[i] += step

            dXdx[:, 2*i : 2*i + 2] = (Rebuild(Perturbed)[:, 1:] - Base[:, 1:])/step

        return dXdx

    def SetSensitivity(self, Rebuild, Variables, step = None) -> None:
        '''
        Parameters
        ----------
        Rebuild : Callable taking a list of design variables and returning the
                  Nodes array the mesher would build from them.
        Variables : Current values of the design variables.
        step : Forward difference step. The default is FDStep.

        Returns
        -------
        None. Sets dXdx and VariableNumber together, so the two can never
        disagree the way they did when each mesher set them by hand.

        '''
        self.dXdx = self._NodeSensitivity(Rebuild, Variables, step = step)
        self.VariableNumber = len(Variables)

        return

    def SetDesignPoints(self, X, Y) -> None:
        '''
        Parameters
        ----------
        X, Y : Co-ordinates of the design points, one per design variable.

        Returns
        -------
        None. Sets VarPos, which the shape design plots overlay on the mesh.

        '''

        self.VarPos = np.vstack((X, Y))

        return

    # ------------------------------------------------------------------
    # Finalisation and validation
    # ------------------------------------------------------------------

    def _Finalise(self) -> None:
        '''
        Returns
        -------
        None. Defaults the load path probe to the largest applied load
        component and checks the mesh. Call this as the last statement of a
        mesher.

        '''
        if self.ProbeDOF is None and np.any(self.Load):

            self.ProbeDOF = int(np.argmax(np.abs(self.Load[:, 0])))

        if self.AutoValidate:

            self.Validate()

        return

    def _ElementAreas(self) -> npt.NDArray[np.float64]:
        '''
        Returns
        -------
        Signed area of each element, from the shoelace formula over its four
        corner nodes, which are the first four columns of the connectivity for
        every element type in the library.
        '''

        Corners = self.Nodes[self.Elements[:, 1:5] - 1, 1:]

        X, Y = Corners[:, :, 0], Corners[:, :, 1]

        return 0.5*np.sum(X*np.roll(Y, -1, axis = 1) - np.roll(X, -1, axis = 1)*Y, axis = 1)

    def Validate(self, Jacobian = False) -> list[str]:
        '''
        Parameters
        ----------
        Jacobian : Also instantiate the element at every element and require a
                   positive detJ at every Gauss point. Off by default because
                   it costs an assembly sized loop; the signed area check below
                   catches the node ordering mistakes that actually happen.

        Returns
        -------
        The warning messages issued, empty if the mesh is clean. Problems that
        would give a wrong answer rather than a suspicious one are collected
        and raised together as a single ValueError.

        '''
        Errors: list[str] = []
        Warnings: list[str] = []

        Errors += self._CheckArrays()

        if Errors:

            # Everything below indexes the arrays that just failed, so stop.
            raise ValueError('This mesh is not valid:\n  ' + '\n  '.join(Errors))

        Errors += self._CheckConnectivity(Jacobian)
        Errors += self._CheckSystem()

        if Errors:

            raise ValueError('This mesh is not valid:\n  ' + '\n  '.join(Errors))

        Warnings += self._CheckSuspicious()

        for Message in Warnings:

            warnings.warn(Message, MeshWarning, stacklevel = 3)

        return Warnings

    def _CheckArrays(self) -> list[str]:
        '''
        Returns
        -------
        Errors about the shape and numbering of Nodes and Elements. Everything
        else indexes these, so they are checked first and on their own.
        '''
        Errors = []

        for Name in ('Nodes', 'Elements', 'DegOfFreedom', 'Load', 'ElementType'):

            if not hasattr(self, Name):

                Errors.append(f'{Name} was never set. A mesher must call SetNodes, SetElements, '
                              'a boundary condition and a load before _Finalise.')

        if Errors:

            return Errors

        if self.Nodes.ndim != 2 or self.Nodes.shape[1] != 3:

            Errors.append(f'Nodes has shape {self.Nodes.shape}, expected (NumNodes, 3) of '
                          '[id, x, y].')

        if self.Elements.ndim != 2:

            Errors.append(f'Elements has shape {self.Elements.shape}, expected '
                          '(NumElements, 1 + NumNodes).')

        if Errors:

            return Errors

        NumNodes = self.Nodes.shape[0]

        if not np.array_equal(self.Nodes[:, 0], np.arange(1, NumNodes + 1)):

            Errors.append('The Nodes id column must run 1 to NumNodes with no gaps and no '
                          'reordering, because the solvers look a node up by its position, '
                          'Nodes[Elements - 1]. A gap gives a wrong answer rather than an error.')

        Expected = self.NumElementNodes

        if self.Elements.shape[1] != 1 + Expected:

            Errors.append(f'Elements has {self.Elements.shape[1]} columns but element type '
                          f'{self.ElementType!r} has {Expected} nodes, so it needs '
                          f'{1 + Expected}: an id column and one column per node.')

        if not np.issubdtype(self.Elements.dtype, np.integer):

            Errors.append(f'Elements has dtype {self.Elements.dtype}, expected an integer type.')

        if not np.array_equal(self.Elements[:, 0], np.arange(1, self.Elements.shape[0] + 1)):

            Errors.append('The Elements id column must run 1 to NumElements with no gaps.')

        return Errors

    def _CheckConnectivity(self, Jacobian) -> list[str]:
        '''
        Parameters
        ----------
        Jacobian : Also check detJ at every Gauss point.

        Returns
        -------
        Errors about the element connectivity itself.
        '''
        Errors = []

        NumNodes = self.Nodes.shape[0]
        Connectivity = self.Elements[:, 1:]

        Bad = np.unique(Connectivity[(Connectivity < 1) | (Connectivity > NumNodes)])

        if Bad.size:

            Errors.append(f'Elements refers to node ids {Bad.tolist()}, which are outside '
                          f'1 to {NumNodes}.')

            return Errors

        Repeated = [int(self.Elements[i, 0]) for i in range(Connectivity.shape[0])
                    if np.unique(Connectivity[i]).size != Connectivity.shape[1]]

        if Repeated:

            Errors.append(f'Elements {Repeated} use the same node more than once, which makes '
                          'them degenerate.')

            return Errors

        # Distinct nodes sitting at the same point are just as degenerate as a
        # repeated id: the edge between them has no length, so the mapping is
        # singular there. A triangle drawn as a quad this way still has area,
        # so the signed area check below does not see it.
        Corners = self.Nodes[self.Elements[:, 1:5] - 1, 1:]
        Scale = max(float(np.max(np.abs(self.Nodes[:, 1:]))), 1.0)

        Gaps = np.linalg.norm(Corners[:, :, None, :] - Corners[:, None, :, :], axis = -1)
        Gaps[:, np.arange(4), np.arange(4)] = np.inf

        Coincident = self.Elements[np.any(Gaps <= self.Tolerance*Scale, axis = (1, 2)), 0]

        if Coincident.size:

            Errors.append(f'Elements {Coincident.tolist()} have two corner nodes at the same '
                          'point, so one of their edges has no length and the mapping is singular '
                          'there.')

            return Errors

        Areas = self._ElementAreas()
        Scale = np.max(np.abs(Areas)) if Areas.size else 0.0

        Collapsed = self.Elements[np.abs(Areas) <= self.Tolerance*max(Scale, 1.0), 0]

        if Collapsed.size:

            Errors.append(f'Elements {Collapsed.tolist()} have no area, so their nodes are '
                          'collinear or coincident.')

        Reversed = self.Elements[Areas < 0, 0]

        if Reversed.size:

            Errors.append(f'Elements {Reversed.tolist()} have their corner nodes in clockwise '
                          'order. The Jacobian is then negative throughout, which flips the sign '
                          'of their contribution to the stiffness and gives a plausible looking '
                          'but wrong answer. Reverse the node order in those rows.')

        if Jacobian and not Errors:

            Errors += self._CheckJacobian()

        return Errors

    def _CheckJacobian(self) -> list[str]:
        '''
        Returns
        -------
        Errors from instantiating the element at every element and asking for
        the Jacobian at each Gauss point. Catches mid side nodes placed badly
        enough to fold the mapping, which the corner area check passes.
        '''
        ElementClass = LookupElement(self.ElementType)

        Bad = []

        for i in range(self.Elements.shape[0]):

            Coordinates = self.Nodes[self.Elements[i, 1:] - 1, 1:]

            Instance = ElementClass(Coordinates, self.t, self.E, self.v, self.plane)

            Points, _ = Instance.GaussPointsAndWeights()

            if any(Instance.detJ(xi, eta) <= 0 for xi in Points for eta in Points):

                Bad.append(int(self.Elements[i, 0]))

        if Bad:

            return [f'Elements {sorted(set(Bad))} have a non positive Jacobian at a Gauss point, '
                    'so the mapping folds over. Check the mid side node positions.']

        return []

    def _CheckSystem(self) -> list[str]:
        '''
        Returns
        -------
        Errors about the free degrees of freedom, the load vector and the
        sensitivities, all of which are sized by the node count.
        '''
        Errors = []

        Size = self.Nodes.shape[0]*2

        DegOfFreedom = np.asarray(self.DegOfFreedom)

        if DegOfFreedom.ndim != 1 or not np.issubdtype(DegOfFreedom.dtype, np.integer):

            Errors.append(f'DegOfFreedom has shape {DegOfFreedom.shape} and dtype '
                          f'{DegOfFreedom.dtype}, expected a one dimensional integer array of the '
                          'free degrees of freedom.')

        elif DegOfFreedom.size == 0:

            Errors.append('DegOfFreedom is empty, so every degree of freedom is held and there is '
                          'nothing to solve for.')

        elif np.unique(DegOfFreedom).size != DegOfFreedom.size:

            Errors.append('DegOfFreedom repeats a degree of freedom, which silently changes the '
                          'size of the system being solved.')

        elif DegOfFreedom.min() < 0 or DegOfFreedom.max() >= Size:

            Errors.append(f'DegOfFreedom runs {DegOfFreedom.min()} to {DegOfFreedom.max()}, '
                          f'outside 0 to {Size - 1}.')

        if np.shape(self.Load) != (Size, 1):

            Errors.append(f'Load has shape {np.shape(self.Load)}, expected ({Size}, 1), two '
                          'degrees of freedom per node.')

        if hasattr(self, 'dXdx'):

            Expected = (self.Nodes.shape[0], 2*getattr(self, 'VariableNumber', 0))

            if np.shape(self.dXdx) != Expected:

                Errors.append(f'dXdx has shape {np.shape(self.dXdx)} but VariableNumber is '
                              f'{getattr(self, "VariableNumber", None)}, so it should be '
                              f'{Expected}. Use SetSensitivity, which sets both together.')

        return Errors

    def _CheckSuspicious(self) -> list[str]:
        '''
        Returns
        -------
        Warnings: things that are legitimate in some meshes but are usually a
        mistake, so they are reported without stopping the build.
        '''
        Messages = []

        Held = self.HeldDOF

        if Held.size < 3 or not np.any(Held % 2 == 0) or not np.any(Held % 2 == 1) \
                or np.unique(Held//2).size < 2:

            Messages.append(f'Only {Held.size} degrees of freedom are held, on '
                            f'{np.unique(Held//2).size} node(s). That usually leaves a rigid body '
                            'mode and a singular stiffness matrix. This is the cheap structural '
                            'check, not a rank test on the assembled matrix, so an unusual set of '
                            'constraints can trip it harmlessly.')

        Lost = np.intersect1d(self.LoadDOF, Held)

        if Lost.size:

            Messages.append(f'Load is applied at degrees of freedom {Lost.tolist()}, which are '
                            'held by a boundary condition. Those components are discarded when '
                            'the system is restricted to the free degrees of freedom, so nothing '
                            'happens there.')

        if not np.any(self.Load):

            Messages.append('Load is all zeros, so every solver returns zero displacement.')

        if self.ProbeDOF is not None and self.ProbeDOF in self._Held:

            Messages.append(f'ProbeDOF {self.ProbeDOF} is held, so its displacement is always '
                            'zero and Plotting.LoadPath draws a flat line. Point it at a free '
                            'degree of freedom with SetProbe.')

        Coordinates = self.Nodes[:, 1:]
        Scale = max(float(np.max(np.abs(Coordinates))), 1.0)

        _, Counts = np.unique(np.round(Coordinates/(self.Tolerance*Scale)), axis = 0,
                              return_counts = True)

        if np.any(Counts > 1):

            Messages.append(f'{int(np.sum(Counts[Counts > 1]))} nodes sit on top of another node. '
                            'That is deliberate for a crack or a hinge and a mistake otherwise.')

        Unused = np.setdiff1d(self.Nodes[:, 0].astype('int'), self.Elements[:, 1:])

        if Unused.size:

            Messages.append(f'Nodes {Unused.tolist()} belong to no element, so they carry no '
                            'stiffness.')

        if hasattr(self, 'VariableNumber') and not hasattr(self, 'dXdx'):

            Messages.append(f'VariableNumber is {self.VariableNumber} but dXdx was never set, so '
                            'any solve with Sensitivity = True will fail. Use SetSensitivity.')

        return Messages
