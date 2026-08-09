"""Abstract base class for every solver in the library.

Solver holds the machinery every solver needs regardless of type: locating the
element class, assembling the global stiffness or tangent, restricting to the
free degrees of freedom, factorising, and differentiating the assembled
quantities with respect to the design variables.

To add a solver, subclass Solver and implement Solve. Results are written onto
the mesh, so the plotting helpers work with any solver.

    class MySolver(Solver):
        def Solve(self):
            K = self._Assemble()
            ...
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import linalg as spla

from fem2d.elements.registry import LookupElement


class Solver(ABC):
    '''
    Base class for the solvers.

    Parameters
    ----------
    Mesh : Mesh object from the Mesher Class.
    Sensitivity : Compute the design sensitivities alongside the solution.
    '''

    def __init__(self, Mesh, Sensitivity: bool = False):

        self.Mesh = Mesh
        self.Sensitivity = Sensitivity

        return

    @abstractmethod
    def Solve(self) -> None:
        '''
        Run the solver, writing the results onto self.Mesh.

        Returns
        -------
        None.
        '''

    def _ElementClass(self, Derivative: bool = False) -> type:
        '''
        Parameters
        ----------
        Derivative : Return the shape derivative class instead of the element.

        Returns
        -------
        The element class named by Mesh.ElementType, which may be a registered
        name or an Element subclass supplied directly.

        '''

        return LookupElement(self.Mesh.ElementType, Derivative = Derivative)

    def _ElementDOF(self, el: int) -> npt.NDArray[np.int64]:
        '''
        Parameters
        ----------
        Mesh : Mesh object from the Mesher Class.
        el : Row index into Mesh.Elements.

        Returns
        -------
        The global degrees of freedom the element spans, in local order.

        '''

        Mesh = self.Mesh
        Local = Mesh.Elements[el, 1:]

        return np.vstack((Local*2 - 2, Local*2 - 1)).T.reshape(-1)

    @staticmethod
    def _Scatter(Triplets, DOF, Matrix):
        '''
        Parameters
        ----------
        Triplets : (Rows, Cols, Vals) lists being accumulated.
        DOF : Global degrees of freedom the element spans.
        Matrix : Element matrix to scatter into the global matrix.

        Returns
        -------
        None. Appends this element's contribution to the triplet lists, which
        is what lets the global matrix be built once in sparse form instead of
        writing into a dense array per element.

        '''
        Rows, Cols, Vals = Triplets

        Rows.append(np.repeat(DOF, DOF.size))
        Cols.append(np.tile(DOF, DOF.size))
        Vals.append(np.asarray(Matrix).reshape(-1))

        return

    @staticmethod
    def _Sparse(Triplets, size):
        '''
        Parameters
        ----------
        Triplets : (Rows, Cols, Vals) lists filled by _Scatter.
        size : Number of degrees of freedom.

        Returns
        -------
        The assembled matrix in CSR form. Duplicate entries are summed, which
        is what performs the assembly.

        '''
        Rows, Cols, Vals = Triplets

        if not Vals:

            return sparse.csr_matrix((size, size))

        return sparse.coo_matrix((np.concatenate(Vals),
                                  (np.concatenate(Rows), np.concatenate(Cols))),
                                 shape = (size, size)).tocsr()

    @staticmethod
    def _Free(K, DegOfFreedom):
        '''
        Parameters
        ----------
        K : Sparse global matrix.
        DegOfFreedom : Unconstrained degrees of freedom.

        Returns
        -------
        The submatrix on the free degrees of freedom. Sparse matrices do not
        take np.ix_, so the rows and columns are sliced in turn.

        '''
        return K[DegOfFreedom,:][:,DegOfFreedom]

    @staticmethod
    def _Factorise(K):
        '''
        Parameters
        ----------
        K : Sparse system matrix.

        Returns
        -------
        An LU factorisation exposing .solve(b). The arc length solver and the
        sensitivity loops reuse one factorisation over many right hand sides,
        so the decomposition is only paid for once.

        '''
        return spla.splu(sparse.csc_matrix(K))

    def _Assemble(self):

        '''
        Parameters
        ----------
        Mesh : Mesh object from the Mesher Class.

        Returns
        -------
        K : Assembles the global stiffness matrix using the problem defined
        in the Mesh object, in sparse CSR form.
        '''

        Mesh = self.Mesh
        Element = self._ElementClass()

        Triplets = ([], [], [])

        for el in range(Mesh.Elements.shape[0]):

            #coordinates in the global stiff matrix
            KCoor = self._ElementDOF(el)

            #Node global coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]

            Ke = Element(NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane).StiffMatrix()

            self._Scatter(Triplets, KCoor, Ke)

        return self._Sparse(Triplets, Mesh.Nodes.shape[0]*2)

    def _MovedNodeDOF(self, var):
        '''
        Parameters
        ----------
        Mesh : Mesh object from Mesher Class.
        var : Index of the design variable.

        Returns
        -------
        Yields (el, LocalDOF, weight) for every element touched by a nodal
        co-ordinate that this design variable moves. weight is dX/dx for that
        co-ordinate, so the caller only has to integrate and scale.

        '''

        Mesh = self.Mesh
        # Derivative of each node coordinate with respect to the design variable
        dXdx = Mesh.dXdx[:, 2*var : 2*var+2].reshape(Mesh.Nodes.shape[0]*2)

        for GlobalDOF in np.flatnonzero(dXdx): #only the coordinates the variable moves

            # Find the node associated with the Global Degree of Freedom
            node = GlobalDOF//2

            # Find the elements that include the node, as well as the local node number
            element, LocalNodes = np.where(Mesh.Elements[:,1:] == node + 1)

            #Transform local node number to local degree of freedom
            LocalDOF = LocalNodes*2 + (GlobalDOF % 2)

            for el, dof in zip(element, LocalDOF, strict=True):

                yield el, dof, dXdx[GlobalDOF]

    def _dKdXVariable(self, var):
        '''
        Parameters
        ----------
        Mesh : Mesh object from Mesher Class.
        var : Index of the design variable.

        Returns
        -------
        dKdx : Derivative of the global stiffness matrix w.r.t the design
               variable, in sparse CSR form, accumulated over every nodal
               co-ordinate the variable moves.

        '''

        Mesh = self.Mesh
        Element = self._ElementClass(Derivative = True)

        Triplets = ([], [], [])

        for el, dof, weight in self._MovedNodeDOF(var):

            KCoor = self._ElementDOF(el)

            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]

            Ke = Element(NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane).Integrate(dof)

            self._Scatter(Triplets, KCoor, Ke * weight)

        return self._Sparse(Triplets, Mesh.Nodes.shape[0]*2)

    def _dRdXVariable(self, var):
        '''
        Parameters
        ----------
        Mesh : Mesh object from Mesher Class.
        var : Index of the design variable.

        Returns
        -------
        dRdx : Derivative of the global residual vector w.r.t the design
               variable, accumulated over every nodal co-ordinate the variable
               moves.

        '''

        Mesh = self.Mesh
        Element = self._ElementClass(Derivative = True)

        dRdx = np.zeros((Mesh.Nodes.shape[0]*2, 1))

        for el, dof, weight in self._MovedNodeDOF(var):

            RCoor = self._ElementDOF(el)

            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]

            dRedX = Element(NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane,
                            LinearFlag = False, U = Mesh.U[RCoor,:]).ResIntegrate(dof)

            dRdx[RCoor,:] += dRedX * weight

        return dRdx

    def _ResAndTangentAssemble(self):
        '''
        Parameters
        ----------
        Mesh : Mesh object form Meshers Class
        U : Current Displacement Estimate.

        Returns
        -------
        Ktangent : Global Stiffness Tangent Matrix, in sparse CSR form.
        GlobalResidual : Global Residual Vector.

        '''

        Mesh = self.Mesh
        Element = self._ElementClass()

        Triplets = ([], [], [])
        GlobalResidual = np.zeros((Mesh.Nodes.shape[0]*2,1))

        for el in range(Mesh.Elements.shape[0]):

            #coordinates in the global stiff matrix
            KCoor = self._ElementDOF(el)

            #Node global coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]

            U = Mesh.U[KCoor,:]

            LocalElementTangent, LocalElementRes = Element(NodeCoor, Mesh.t, Mesh.E,
                                                           Mesh.v, Mesh.plane,
                                                           LinearFlag = False, U = U).ResTangent()

            self._Scatter(Triplets, KCoor, LocalElementTangent)
            GlobalResidual[KCoor, :] += LocalElementRes

        return self._Sparse(Triplets, Mesh.Nodes.shape[0]*2), GlobalResidual

