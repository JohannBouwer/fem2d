"""Solvers.

FEMSolvers is kept as a thin façade over the solver classes so existing call
sites keep working; new code can use the classes directly.
"""

from fem2d.solvers.arclength import ArcLengthSolver

# Aliased so the façade methods below can share the solvers' names.
from fem2d.solvers.arclength import ArcLengthSolver as _ArcLength
from fem2d.solvers.base import Solver
from fem2d.solvers.linear import LinearSolver
from fem2d.solvers.linear import LinearSolver as _Linear
from fem2d.solvers.newton import NonLinearSolver
from fem2d.solvers.newton import NonLinearSolver as _NonLinear

__all__ = ['Solver', 'LinearSolver', 'NonLinearSolver', 'ArcLengthSolver', 'FEMSolvers']


class FEMSolvers:
    '''
    Function style façade over the solver classes.

    Each method builds the corresponding solver and runs it, writing the
    results onto the mesh exactly as before.
    '''

    @staticmethod
    def LinearSolver(Mesh, Sensitivity = False):
        '''Run LinearSolver on Mesh.'''

        return _Linear(Mesh, Sensitivity = Sensitivity).Solve()

    @staticmethod
    def NonLinearSolver(Mesh, LoadSteps = 1, MaxIter = 8, tol = 1e-4, Sensitivity = False):
        '''Run NonLinearSolver on Mesh.'''

        return _NonLinear(Mesh, LoadSteps = LoadSteps, MaxIter = MaxIter,
                          tol = tol, Sensitivity = Sensitivity).Solve()

    @staticmethod
    def ArcLengthSolver(Mesh, ArcLength, TotalArcLength, psi = 0, tol = 1e-4,
                        MaxIter = 8, Sensitivity = False, MaxCuts = 12):
        '''Run ArcLengthSolver on Mesh.'''

        return _ArcLength(Mesh, ArcLength, TotalArcLength, psi = psi, tol = tol,
                          MaxIter = MaxIter, Sensitivity = Sensitivity,
                          MaxCuts = MaxCuts).Solve()
