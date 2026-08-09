"""Linear solver."""

import numpy as np

from fem2d.solvers.base import Solver


class LinearSolver(Solver):
    '''
    Small deformation elastic solve.

    Parameters
    ----------
    Mesh : Mesh object from the Mesher Class.
    Sensitivity : Compute dUdx alongside the solution.
    '''

    def Solve(self):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the Mesher Class.

        Returns
        -------
        U : Full Resultant Displacement Vector.

        '''

        Mesh = self.Mesh
        Sensitivity = self.Sensitivity
        Mesh.KFull = self._Assemble()
        Mesh.U = np.zeros((Mesh.KFull.shape[0], 1))
        
        KFree = self._Free(Mesh.KFull, Mesh.DegOfFreedom)
        Mesh.K = KFree

        # One factorisation serves the solve and every design variable.
        Factor = self._Factorise(KFree)

        Mesh.U[Mesh.DegOfFreedom,:] = Factor.solve(Mesh.Load[Mesh.DegOfFreedom])
        Mesh.AllU = Mesh.U

        if Sensitivity:

            Mesh.dUdx = np.zeros((Mesh.U.shape[0], Mesh.VariableNumber))

            for var in range(Mesh.VariableNumber):

                dKdx = self._Free(self._dKdXVariable(var), Mesh.DegOfFreedom)

                dUdX = Factor.solve(-1*(dKdx @ Mesh.U[Mesh.DegOfFreedom]))

                Mesh.dUdx[Mesh.DegOfFreedom, var] += dUdX[:,0]

        return
