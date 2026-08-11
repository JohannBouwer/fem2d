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

    def Solve(self) -> None:
        '''
        Solve K U = F once and write U, K, KFull and AllU onto the mesh.

        Returns
        -------
        None. With Sensitivity on, Mesh.dUdx is written too.
        '''

        Mesh = self.Mesh
        Sensitivity = self.Sensitivity
        # Records which strain measure the result belongs to, so post processing
        # knows whether to read it as small strain or finite strain.
        Mesh.LargeDeflection = False

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

            # Every design variable at once: the element integrals are shared
            # between them, so this costs what one variable used to.
            dKdxAll = self._dKdXAll()

            for var in range(Mesh.VariableNumber):

                dKdx = self._Free(dKdxAll[var], Mesh.DegOfFreedom)

                dUdX = Factor.solve(-1*(dKdx @ Mesh.U[Mesh.DegOfFreedom]))

                Mesh.dUdx[Mesh.DegOfFreedom, var] += dUdX[:,0]

        return
