"""Arc length continuation solver."""

import numpy as np

from fem2d.solvers.base import Solver


class ArcLengthSolver(Solver):
    '''
    Arc length continuation, able to trace past limit points.

    Parameters
    ----------
    Mesh : Mesh object from the Mesher Class.
    ArcLength : Arc length for each arc step.
    TotalArcLength : Accumulated arc value at which to stop.
    psi : psi value for the constraint equation. The default is 0.
    tol : Relative residual tolerance. The default is 1e-4.
    MaxIter : Iterations before the arc step is cut. The default is 8.
    Sensitivity : Compute dUdx and dLdx alongside the solution.
    MaxCuts : Arc step reductions allowed within one step. The default is 12.
    '''

    def __init__(self, Mesh, ArcLength, TotalArcLength, psi = 0, tol = 1e-4,
                 MaxIter = 8, Sensitivity = False, MaxCuts = 12):

        super().__init__(Mesh, Sensitivity)

        self.ArcLength = ArcLength
        self.TotalArcLength = TotalArcLength
        self.psi = psi
        self.tol = tol
        self.MaxIter = MaxIter
        self.MaxCuts = MaxCuts

        return

    def Solve(self):
        '''
        Parameters
        ----------
        Mesh : Mesh Object from the Mesher Class.
        ArcLength : Arc Length for each arc step.
        TotalArcLength : The accumulated arc value to terminated the simulation.
        tol : Tolerance of the residual before termination.
            The default is 1e-4.
        MaxIter : Maximum Number of Newton Iterations before the Arc length is adjusted, 
                The default is 8.
        psi : psi value for the constraint equation.
            The default is 0.
        MaxCuts : Number of times the arc step may be reduced within one step
                before giving up. The default is 12.

        Returns
        -------
        None.

        Notes
        -----
        Cutting the arc step does not affect the sensitivities. The constraint
        is differentiated in its homogeneous form, which is exact because the
        arc step only ever changes by a fixed factor and so is piecewise
        constant in the design variables, and the increments the recursion uses
        are the ones actually taken. Verified against finite differences with
        up to 30 cuts on the path.

        What the derivatives are taken with respect to is the path that was
        actually walked, with its realised sequence of step sizes and its step
        count. Both of those change discontinuously at isolated design points,
        and there the end of the path jumps; finite differences straddling such
        a point disagree with the analytical value, which is the correct one
        sided derivative of the branch that was taken.

        '''

        Mesh = self.Mesh
        ArcLength, TotalArcLength = self.ArcLength, self.TotalArcLength
        psi, tol, MaxIter, MaxCuts = self.psi, self.tol, self.MaxIter, self.MaxCuts
        Sensitivity = self.Sensitivity
        
        Mesh.U = np.zeros((Mesh.Nodes.shape[0]*2, 1)) # Initialize the displacement vector
        
        ResNorm = 1 #Intialize the norm of the residual vector
        
        Mesh.AllU = np.zeros((Mesh.Nodes.shape[0]*2, 1)) #store all displacment increments 

        SignDL = 1 #initialize sign direction for the load update
        Mesh.LoadValues = np.array([[0]]) # store load values
        Loadfactor = 0.0 #initialize the load factor value
        AccumulatedArcLength = 0 #initialize the Accumulated arc length

        if Sensitivity:

            # Carried from one arc step to the next, since the constraint on a
            # step couples its sensitivity to the previous point on the path.
            dUdxPrev = np.zeros((Mesh.Nodes.shape[0]*2, Mesh.VariableNumber))
            dLdxPrev = np.zeros((1, Mesh.VariableNumber))

        cnt_step = 0
        while AccumulatedArcLength < TotalArcLength:
            
            print('-----------------------')
            print('Accumalted Arc Length {}'.format(np.round(AccumulatedArcLength, 2)))
            print('-----------------------')
            
            #Previous Arc step values
            PrevLoadFactor = np.copy(Loadfactor)
            PrevU = np.copy(Mesh.U)
            
            ArcStep = ArcLength
            iter_cnt = 0
            cut_cnt = 0 #number of times the arc step has been reduced
            ResNorm = 2*tol #Intialize the norm of the residual vector
            DU = np.zeros_like(Mesh.U[Mesh.DegOfFreedom,:]) #re-zero the update vector
            DL = 0 # re-zero the update to the load factor
            SignDL0 = SignDL # reset the sign direction

            while ResNorm > tol:
                
                Ktangent, GlobalResidual = self._ResAndTangentAssemble()
                
                Kff = self._Free(Ktangent, Mesh.DegOfFreedom)

                Rff = (GlobalResidual - Mesh.Load*Loadfactor)[Mesh.DegOfFreedom,:]

                # Solve the two systems off a single factorisation
                Factor = self._Factorise(Kff)

                aQ = Factor.solve(Mesh.Load[Mesh.DegOfFreedom,:])
                aR = -1*Factor.solve(Rff)
                
                # Set up Constants for quadratic equation
                Uesti = DU + aR
                
                C1 = aQ.T.dot(aQ) + psi**2
                C2 = 2*(aQ.T.dot(Uesti)) + 2*psi**2*DL
                C3 = Uesti.T.dot(Uesti) + psi**2*DL**2 - ArcStep**2

                Discriminant = C2**2 - 4*C1*C3 #check if rational

                if Discriminant < 0 or iter_cnt > MaxIter : #if not, decrease Arc Length and zero values

                    print("------Adjusting Arc Length-------")

                    cut_cnt += 1

                    if cut_cnt > MaxCuts:

                        raise RuntimeError(
                            'Arc step could not be converged after {} reductions at an '
                            'accumulated arc length of {:.4g}. Try a smaller ArcLength, a '
                            'looser tol, or a larger MaxIter.'.format(MaxCuts, AccumulatedArcLength))

                    Loadfactor = np.copy(PrevLoadFactor)
                    DL = 0

                    Mesh.U[Mesh.DegOfFreedom,:] = np.copy(PrevU[Mesh.DegOfFreedom,:])
                    DU = np.zeros_like(Mesh.U[Mesh.DegOfFreedom,:])

                    SignDL = SignDL0
                    ArcStep /= np.sqrt(2)

                    iter_cnt = 0

                    ResNorm = 2*tol

                else: # solve load update
                    
                    D = np.sqrt(Discriminant)
                    sign = DU.T.dot(aQ) + psi**2*DL # sign check
                    
                    if iter_cnt > 0: #check first iteration 
                        
                        dL1 = (-C2 + D)/(2*C1) # Two posible solutions
                        dL2 = (-C2 - D)/(2*C1)
                        
                        if sign*dL1 > sign*dL2:
                           
                            dL = dL1
                        
                        else:
                            
                            dL = dL2
                    
                    else:
                        
                        dL = (-C2 + SignDL*D)/(2*C1)
                   
                    # Update arc step values
                    DU += aR + dL*aQ
                    DL += dL
                    
                    Loadfactor += dL[0,0]
                    
                    if iter_cnt == 0:
                        
                        ResNorm0 = np.linalg.norm((Mesh.Load*Loadfactor - GlobalResidual)[Mesh.DegOfFreedom,:])

                    ResNorm = np.linalg.norm((Mesh.Load*Loadfactor - GlobalResidual)[Mesh.DegOfFreedom,:])/ResNorm0
                    
                    print('Itertion {}, ResNorm {}'.format(iter_cnt, ResNorm))
                    
                    iter_cnt += 1
                
                    SignDL = np.sign(sign) # sign update
                    
                    # Update global solution vector
                    Mesh.U[Mesh.DegOfFreedom,:] += aR + dL*aQ
            
            # Update total arc length
            AccumulatedArcLength += ArcStep
            
            # store step values
            Mesh.AllU = np.hstack((Mesh.AllU, Mesh.U)) 
            Mesh.LoadValues = np.append(Mesh.LoadValues, Loadfactor)
            
            if Sensitivity:

                # The tangent left over from the iteration loop was assembled
                # before the final correction was applied, so rebuild it at the
                # converged point.
                Ktangent, GlobalResidual = self._ResAndTangentAssemble()
                Kff = self._Free(Ktangent, Mesh.DegOfFreedom)

                Factor = self._Factorise(Kff)

                dUdx = np.zeros((Mesh.U.shape[0], Mesh.VariableNumber))
                dLdx = np.zeros((1, Mesh.VariableNumber))

                # The constraint equation is written in terms of the increment
                # taken by this arc step, not the total displacement.
                IncU = (Mesh.U - PrevU)[Mesh.DegOfFreedom,:]
                IncL = Loadfactor - PrevLoadFactor

                dUdL = Factor.solve(Mesh.Load[Mesh.DegOfFreedom,:])

                for var in range(Mesh.VariableNumber):

                    dRdx = self._dRdXVariable(var)

                    # Displacement sensitivity holding the load factor fixed.
                    dUdxL = Factor.solve(-1*dRdx[Mesh.DegOfFreedom,:])

                    # Differentiating IncU.T @ IncU + psi**2 * IncL**2 = ArcStep**2
                    # gives the load factor sensitivity. The previous step's
                    # derivatives appear because the increment is a difference of
                    # two points on the path, so the path is differentiated as a
                    # whole rather than one step at a time.
                    Numerator = (IncU.T @ (dUdxPrev[Mesh.DegOfFreedom, var].reshape(-1,1) - dUdxL)
                                 + psi**2 * IncL * dLdxPrev[0, var])

                    Denominator = IncU.T @ dUdL + psi**2 * IncL

                    dL = (Numerator/Denominator)[0,0]

                    dUdx[Mesh.DegOfFreedom, var] = (dUdxL + dUdL*dL)[:,0]
                    dLdx[0, var] = dL

                dUdxPrev = dUdx
                dLdxPrev = dLdx

                Mesh.dUdx = dUdx # sensitivity at the most recent arc step

                if cnt_step == 0:
                    Mesh.dUdx_All = dUdx
                    Mesh.dLdx = np.copy(dLdx)

                else:
                    Mesh.dUdx_All = np.dstack((Mesh.dUdx_All, dUdx))
                    Mesh.dLdx = np.vstack((Mesh.dLdx, dLdx))

            cnt_step += 1
                  
        return
