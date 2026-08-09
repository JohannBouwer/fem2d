import warnings

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from fem2d.elements import FiveBeta, Q4, Q8, d5BdX, dQ4dX, dQ8dX

class FEMSolvers(object):
    
    def _ElementClass(Mesh, Derivative = False):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the Mesher Class.
        Derivative : Return the shape derivitive class instead of the element.

        Returns
        -------
        The element class named by Mesh.ElementType.

        '''
        Types = {'Q4' : (Q4, dQ4dX),
                 '5B' : (FiveBeta, d5BdX),
                 'Q8' : (Q8, dQ8dX)}

        if Mesh.ElementType not in Types:

            raise ValueError('Unknown ElementType {!r}, expected one of {}'.format(
                             Mesh.ElementType, sorted(Types)))

        return Types[Mesh.ElementType][1 if Derivative else 0]

    def _ElementDOF(Mesh, el):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the Mesher Class.
        el : Row index into Mesh.Elements.

        Returns
        -------
        The global degrees of freedom the element spans, in local order.

        '''
        Local = Mesh.Elements[el, 1:]

        return np.vstack((Local*2 - 2, Local*2 - 1)).T.reshape(-1)

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

    def _Assemble(Mesh):

        '''
        Parameters
        ----------
        Mesh : Mesh object from the Mesher Class.

        Returns
        -------
        K : Assembles the global stiffness matrix using the problem defined
        in the Mesh object, in sparse CSR form.
        '''
        Element = FEMSolvers._ElementClass(Mesh)

        Triplets = ([], [], [])

        for el in range(Mesh.Elements.shape[0]):

            #coordinates in the gloabl stiff matrix
            KCoor = FEMSolvers._ElementDOF(Mesh, el)

            #Node gloabl coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]

            Ke = Element(NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane).StiffMatrix()

            FEMSolvers._Scatter(Triplets, KCoor, Ke)

        return FEMSolvers._Sparse(Triplets, Mesh.Nodes.shape[0]*2)

    def _MovedNodeDOF(Mesh, var):
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
        dXdx = Mesh.dXdx[:, 2*var : 2*var+2].reshape(Mesh.Nodes.shape[0]*2) # Derivitive of node coordinate w.r.t the design variable

        for GlobalDOF in np.flatnonzero(dXdx): #only the coordinates the variable moves

            # Find the node associated with the Global Degree of Freedom
            node = GlobalDOF//2

            # Find the elements that include the node, as well as the local node number
            element, LocalNodes = np.where(Mesh.Elements[:,1:] == node + 1)

            #Transform local node number to local degree of freedom
            LocalDOF = LocalNodes*2 + (GlobalDOF % 2)

            for el, dof in zip(element, LocalDOF):

                yield el, dof, dXdx[GlobalDOF]

    def _dKdXVariable(Mesh, var):
        '''
        Parameters
        ----------
        Mesh : Mesh object from Mesher Class.
        var : Index of the design variable.

        Returns
        -------
        dKdx : Derivitive of the global stiffness matrix w.r.t the design
               variable, in sparse CSR form, accumulated over every nodal
               co-ordinate the variable moves.

        '''
        Element = FEMSolvers._ElementClass(Mesh, Derivative = True)

        Triplets = ([], [], [])

        for el, dof, weight in FEMSolvers._MovedNodeDOF(Mesh, var):

            KCoor = FEMSolvers._ElementDOF(Mesh, el)

            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]

            Ke = Element(NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane).Integrate(dof)

            FEMSolvers._Scatter(Triplets, KCoor, Ke * weight)

        return FEMSolvers._Sparse(Triplets, Mesh.Nodes.shape[0]*2)

    def _dRdXVariable(Mesh, var):
        '''
        Parameters
        ----------
        Mesh : Mesh object from Mesher Class.
        var : Index of the design variable.

        Returns
        -------
        dRdx : Derivitive of the global residual vector w.r.t the design
               variable, accumulated over every nodal co-ordinate the variable
               moves.

        '''
        Element = FEMSolvers._ElementClass(Mesh, Derivative = True)

        dRdx = np.zeros((Mesh.Nodes.shape[0]*2, 1))

        for el, dof, weight in FEMSolvers._MovedNodeDOF(Mesh, var):

            RCoor = FEMSolvers._ElementDOF(Mesh, el)

            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]

            dRedX = Element(NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane,
                            LinearFlag = False, U = Mesh.U[RCoor,:]).ResIntegrate(dof)

            dRdx[RCoor,:] += dRedX * weight

        return dRdx

    def LinearSolver(Mesh, Sensitivity = False):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the Mesher Class.

        Returns
        -------
        U : Full Resultant Displacment Vector.

        '''
        Mesh.KFull = FEMSolvers._Assemble(Mesh)
        Mesh.U = np.zeros((Mesh.KFull.shape[0], 1))
        
        KFree = FEMSolvers._Free(Mesh.KFull, Mesh.DegOfFreedom)
        Mesh.K = KFree

        # One factorisation serves the solve and every design variable.
        Factor = FEMSolvers._Factorise(KFree)

        Mesh.U[Mesh.DegOfFreedom,:] = Factor.solve(Mesh.Load[Mesh.DegOfFreedom])
        Mesh.AllU = Mesh.U

        if Sensitivity:

            Mesh.dUdx = np.zeros((Mesh.U.shape[0], Mesh.VariableNumber))

            for var in range(Mesh.VariableNumber):

                dKdx = FEMSolvers._Free(FEMSolvers._dKdXVariable(Mesh, var), Mesh.DegOfFreedom)

                dUdX = Factor.solve(-1*(dKdx @ Mesh.U[Mesh.DegOfFreedom]))

                Mesh.dUdx[Mesh.DegOfFreedom, var] += dUdX[:,0]

        return 
    
    def _ResAndTangentAssemble(Mesh):
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
        Element = FEMSolvers._ElementClass(Mesh)

        Triplets = ([], [], [])
        GlobalResidual = np.zeros((Mesh.Nodes.shape[0]*2,1))

        for el in range(Mesh.Elements.shape[0]):

            #coordinates in the gloabl stiff matrix
            KCoor = FEMSolvers._ElementDOF(Mesh, el)

            #Node gloabl coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]

            U = Mesh.U[KCoor,:]

            LocalElementTangent, LocalElementRes = Element(NodeCoor, Mesh.t, Mesh.E,
                                                           Mesh.v, Mesh.plane,
                                                           LinearFlag = False, U = U).ResTangent()

            FEMSolvers._Scatter(Triplets, KCoor, LocalElementTangent)
            GlobalResidual[KCoor, :] += LocalElementRes

        return FEMSolvers._Sparse(Triplets, Mesh.Nodes.shape[0]*2), GlobalResidual

    def NonLinearSolver(Mesh, LoadSteps = 1, MaxIter = 8, tol = 1e-4, Sensitivity = False):
        '''
        Parameters
        ----------
        Mesh : Mesh object from Mesher Class.
        MaxIter : Maximum Number of Newton Iterations,
                The default is 8.
        tol : Tolerance of the residual before termination.
            The default is 1e-4.

        Returns
        -------
        U : Final Displacement Vector.

        Warns
        -----
        RuntimeWarning if a load step exhausts MaxIter without reaching tol.
        The displacements are still returned, but they are not an equilibrium
        point, so any sensitivity computed from them is meaningless.

        '''
        Mesh.U = np.zeros((Mesh.Nodes.shape[0]*2, 1)) # Initialize the displacement vector
        
        ResNorm = 1 #Intialize the norm of the residual vector
        
        Mesh.AllU = np.zeros((Mesh.Nodes.shape[0]*2, LoadSteps + 1)) #store all displacment increments 
        
        LoadStep = Mesh.Load/LoadSteps
        Mesh.LoadValues = np.linspace(0, 1, LoadSteps + 1)
        for i in range(LoadSteps):
            
            print('---------------')
            print('Load Step {}'.format(i + 1))
            print('---------------')
            
            Mesh.AllU[:, [i + 1]] += Mesh.U
            
            iter_cnt = 0
            ResNorm = 1 #Intialize the norm of the residual vector
            while ResNorm > tol and iter_cnt < MaxIter:
                
                Ktangent, GlobalResidual = FEMSolvers._ResAndTangentAssemble(Mesh)
                Mesh.K = Ktangent
                Kff = FEMSolvers._Free(Ktangent, Mesh.DegOfFreedom)
                Rff = (GlobalResidual - LoadStep*(i+1))[Mesh.DegOfFreedom,:]

                Uff = -1*FEMSolvers._Factorise(Kff).solve(Rff)
                
                Mesh.AllU[Mesh.DegOfFreedom, i + 1] += Uff[:,0]
                
                Mesh.U[Mesh.DegOfFreedom,:] += Uff
                
                if iter_cnt == 0:
                    
                    ResNorm0 = np.linalg.norm(Rff)
                
                ResNorm = np.linalg.norm(Rff)/ResNorm0
                
                print('Interation {}, Residual Norm {}'.format(iter_cnt, ResNorm))

                iter_cnt += 1

            if ResNorm > tol:

                warnings.warn('Load step {} stopped at {} iterations with a '
                              'residual norm of {:.3e}, above the tolerance of {:.3e}. '
                              'The result is not an equilibrium point.'.format(i + 1, MaxIter, ResNorm, tol),
                              RuntimeWarning, stacklevel = 2)

        if Sensitivity:

            # The tangent left over from the iteration loop was assembled before
            # the final correction was applied, so rebuild it at the converged point.
            Ktangent, GlobalResidual = FEMSolvers._ResAndTangentAssemble(Mesh)
            Kff = FEMSolvers._Free(Ktangent, Mesh.DegOfFreedom)

            Factor = FEMSolvers._Factorise(Kff)

            Mesh.dUdx = np.zeros((Mesh.U.shape[0], Mesh.VariableNumber))

            for var in range(Mesh.VariableNumber):

                dRdx = FEMSolvers._dRdXVariable(Mesh, var)

                dUdX = Factor.solve(-1*dRdx[Mesh.DegOfFreedom,:])

                Mesh.dUdx[Mesh.DegOfFreedom, var] += dUdX[:,0]
            
        return 
    
    def ArcLengthSolver(Mesh, ArcLength, TotalArcLength, psi = 0, tol = 1e-4, MaxIter = 8, Sensitivity = False, MaxCuts = 12):
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
                
                Ktangent, GlobalResidual = FEMSolvers._ResAndTangentAssemble(Mesh)
                
                Kff = FEMSolvers._Free(Ktangent, Mesh.DegOfFreedom)

                Rff = (GlobalResidual - Mesh.Load*Loadfactor)[Mesh.DegOfFreedom,:]

                # Solve the two systems off a single factorisation
                Factor = FEMSolvers._Factorise(Kff)

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
                Ktangent, GlobalResidual = FEMSolvers._ResAndTangentAssemble(Mesh)
                Kff = FEMSolvers._Free(Ktangent, Mesh.DegOfFreedom)

                Factor = FEMSolvers._Factorise(Kff)

                dUdx = np.zeros((Mesh.U.shape[0], Mesh.VariableNumber))
                dLdx = np.zeros((1, Mesh.VariableNumber))

                # The constraint equation is written in terms of the increment
                # taken by this arc step, not the total displacement.
                IncU = (Mesh.U - PrevU)[Mesh.DegOfFreedom,:]
                IncL = Loadfactor - PrevLoadFactor

                dUdL = Factor.solve(Mesh.Load[Mesh.DegOfFreedom,:])

                for var in range(Mesh.VariableNumber):

                    dRdx = FEMSolvers._dRdXVariable(Mesh, var)

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

            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
