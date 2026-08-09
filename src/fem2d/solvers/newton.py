"""Newton solver with the load applied in steps."""

import logging
import warnings

import numpy as np

from fem2d.solvers.base import Solver

logger = logging.getLogger(__name__)


class NonLinearSolver(Solver):
    '''
    Large deformation solve, Newton iterations within each load step.

    Parameters
    ----------
    Mesh : Mesh object from the Mesher Class.
    LoadSteps : Number of equal load increments. The default is 1.
    MaxIter : Maximum Newton iterations per load step. The default is 8.
    tol : Relative residual tolerance. The default is 1e-4.
    Sensitivity : Compute dUdx alongside the solution.
    '''

    def __init__(self, Mesh, LoadSteps: int = 1, MaxIter: int = 8, tol: float = 1e-4,
                 Sensitivity: bool = False):

        super().__init__(Mesh, Sensitivity)

        self.LoadSteps = LoadSteps
        self.MaxIter = MaxIter
        self.tol = tol

        return

    def Solve(self) -> None:
        '''
        Apply the load in equal steps, iterating to equilibrium within each.

        Returns
        -------
        None. Writes U, AllU and LoadValues onto the mesh, and dUdx with
        Sensitivity on.

        Warns
        -----
        RuntimeWarning if a load step exhausts MaxIter without reaching tol.
        The displacements are still returned, but they are not an equilibrium
        point, so any sensitivity computed from them is meaningless.
        '''

        Mesh = self.Mesh
        LoadSteps, MaxIter, tol = self.LoadSteps, self.MaxIter, self.tol
        Sensitivity = self.Sensitivity
        # Records which strain measure the result belongs to, so post processing
        # knows whether to read it as small strain or finite strain.
        Mesh.LargeDeflection = True

        Mesh.U = np.zeros((Mesh.Nodes.shape[0]*2, 1)) # Initialize the displacement vector

        ResNorm = 1 #Initialize the norm of the residual vector

        Mesh.AllU = np.zeros((Mesh.Nodes.shape[0]*2, LoadSteps + 1)) #store all displacement increments

        LoadStep = Mesh.Load/LoadSteps
        Mesh.LoadValues = np.linspace(0, 1, LoadSteps + 1)
        for i in range(LoadSteps):

            logger.info('Load step %d of %d', i + 1, LoadSteps)

            Mesh.AllU[:, [i + 1]] += Mesh.U

            iter_cnt = 0
            ResNorm = 1 #Initialize the norm of the residual vector
            while ResNorm > tol and iter_cnt < MaxIter:

                Ktangent, GlobalResidual = self._ResAndTangentAssemble()
                Mesh.K = Ktangent
                Kff = self._Free(Ktangent, Mesh.DegOfFreedom)
                Rff = (GlobalResidual - LoadStep*(i+1))[Mesh.DegOfFreedom,:]

                Uff = -1*self._Factorise(Kff).solve(Rff)

                Mesh.AllU[Mesh.DegOfFreedom, i + 1] += Uff[:,0]

                Mesh.U[Mesh.DegOfFreedom,:] += Uff

                if iter_cnt == 0:

                    ResNorm0 = np.linalg.norm(Rff)

                ResNorm = np.linalg.norm(Rff)/ResNorm0

                logger.info('  iteration %d, residual norm %.6e', iter_cnt, ResNorm)

                iter_cnt += 1

            if ResNorm > tol:

                warnings.warn(f'Load step {i + 1} stopped at {MaxIter} iterations with a '
                              f'residual norm of {ResNorm:.3e}, above the tolerance of {tol:.3e}. '
                              'The result is not an equilibrium point.',
                              RuntimeWarning, stacklevel = 2)

        if Sensitivity:

            # The tangent left over from the iteration loop was assembled before
            # the final correction was applied, so rebuild it at the converged point.
            Ktangent, GlobalResidual = self._ResAndTangentAssemble()
            Kff = self._Free(Ktangent, Mesh.DegOfFreedom)

            Factor = self._Factorise(Kff)

            Mesh.dUdx = np.zeros((Mesh.U.shape[0], Mesh.VariableNumber))

            for var in range(Mesh.VariableNumber):

                dRdx = self._dRdXVariable(var)

                dUdX = Factor.solve(-1*dRdx[Mesh.DegOfFreedom,:])

                Mesh.dUdx[Mesh.DegOfFreedom, var] += dUdX[:,0]

        return
