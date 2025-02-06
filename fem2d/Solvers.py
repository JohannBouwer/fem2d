import numpy as np
from Elements4Node import *
from Elements8Node import *
from Sensitivity import *

class FEMSolvers(object):
    
    def _Assemble(Mesh):
        
        '''
        Parameters
        ----------
        Mesh : Mesh object from the Mesher Class.
        
        Returns
        -------
        K : Assembles the global stiffness matrix using the problem defined
        in the Mesh object.
        '''
        if Mesh.ElementType == 'Q4':
            
            Element = Q4
            
        if Mesh.ElementType == '5B':
            
            Element = FiveBeta
        
        if Mesh.ElementType == 'Q8':
            
            Element = Q8
            
        K = np.zeros((Mesh.Nodes.shape[0]*2, Mesh.Nodes.shape[0]*2))

        for el in range(Mesh.Elements.shape[0]):
            

            #Node numbers in the element
            Local = Mesh.Elements[el,1:].reshape(-1,1)
            
            #coordinates in the gloabl stiff matrix
            KCoor = np.hstack((Local*2 - 2, Local*2 - 1))
            KCoor = KCoor.reshape(KCoor.size, )
            
            #Node gloabl coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]
            
            Ke = Element(NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane).StiffMatrix()
           
            K[np.ix_(KCoor, KCoor)] += Ke
            
        return K
    
    def _dKedX(Mesh, DOF):
        '''
        Parameters
        ----------
        Mesh : Mesh object from Mesher Class.
        DOF : Global Degree of Freedom to take the derivitive w.r.t.

        Returns
        -------
        K : The dervivitve Stiffness matrix.

        '''
        if Mesh.ElementType == 'Q4':
            
            Element = dQ4dX
            
        if Mesh.ElementType == '5B':
            
            Element = d5BdX
        
        if Mesh.ElementType == 'Q8':
            
            Element = dQ8dX
            
        K = np.zeros((Mesh.Nodes.shape[0]*2, Mesh.Nodes.shape[0]*2))
        
        # Find the node associated with the Global Degree of Freedom
        node = DOF//2
        # Find the elements that include the nodes, as well as the local node number
        element, LocalNodes = np.where(Mesh.Elements[:,1:] == node + 1)
        
        #Transform local node numbner to local degree of freedom
        if DOF%2:
            
            LocalDOF = LocalNodes*2 + 1
            
        else:
            
            LocalDOF = LocalNodes*2 
       
        for el, dof in zip(element, LocalDOF):
 
            #Node gloabl coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]
            
            #Node numbers in the element
            Local = Mesh.Elements[el,1:].reshape(-1,1)
            
            #coordinates in the gloabl stiff matrix
            KCoor = np.hstack((Local*2 - 2, Local*2 - 1))
            KCoor = KCoor.reshape(KCoor.size, )
            
            #Node gloabl coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]
            
            Ke = Element(NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane).Integrate(dof)
           
            K[np.ix_(KCoor, KCoor)] += Ke
        
        return K
    
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
        
        KFree = Mesh.KFull[np.ix_(Mesh.DegOfFreedom, Mesh.DegOfFreedom)]
        Mesh.K = KFree
        Mesh.U[Mesh.DegOfFreedom,:] = np.linalg.solve(KFree, Mesh.Load[Mesh.DegOfFreedom])
        Mesh.AllU = Mesh.U
        
        if Sensitivity:
            
            Mesh.dUdx = np.zeros((Mesh.U.shape[0], Mesh.VariableNumber))
            
            for var in range(Mesh.VariableNumber):
            
                DOF = np.arange(0, Mesh.Nodes[-1,0].astype('int')*2, 1) # All Global degree of freedom vector
                dXdx = Mesh.dXdx[:, 2*var : 2*var+2].reshape(Mesh.U.shape[0]) # Derivitive of node coordinate w.r.t the design variable
                DOF = DOF[dXdx != 0] #remove zero terms
                dKdx = np.zeros_like(Mesh.KFull)
                
                for dof in DOF:
                    
                    dKdx += FEMSolvers._dKedX(Mesh, dof) * dXdx[dof]
                    
                dKdx = dKdx[np.ix_(Mesh.DegOfFreedom, Mesh.DegOfFreedom)]
                
                dUdX = np.linalg.solve(KFree, -1*dKdx @ Mesh.U[Mesh.DegOfFreedom])
                
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
        Ktangent : Global Stiffness Tangent Matrix.
        GlobalResidual : Global Residual Vector.

        '''
        if Mesh.ElementType == 'Q4':
            
            Element = Q4
            
        if Mesh.ElementType == '5B':
            
            Element = FiveBeta
        
        if Mesh.ElementType == 'Q8':
            
            Element = Q8
            
        Ktangent = np.zeros((Mesh.Nodes.shape[0]*2, Mesh.Nodes.shape[0]*2))
        GlobalResidual = np.zeros((Mesh.Nodes.shape[0]*2,1))
        
        for el in range(Mesh.Elements.shape[0]):
            
            #Node gloabl coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]
            
            #Node numbers in the element
            Local = Mesh.Elements[el,1:].reshape(-1,1)
            
            #coordinates in the gloabl stiff matrix
            KCoor = np.hstack((Local*2 - 2, Local*2 - 1))
            KCoor = KCoor.reshape(KCoor.size, )
            
            #Node gloabl coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]
            U = Mesh.U[KCoor,:]
            
            LocalElementTangent, LocalElementRes = Element(NodeCoor, Mesh.t, Mesh.E, 
                                                           Mesh.v, Mesh.plane, 
                                                           LinearFlag = False, U = U).ResTangent()
           
            Ktangent[np.ix_(KCoor, KCoor)] += LocalElementTangent
            GlobalResidual[KCoor, :] += LocalElementRes
            
        return Ktangent, GlobalResidual
    
    def _dRdX(Mesh, DOF):
        '''
        Parameters
        ----------
        Mesh : Mesh object from Mesher Class.
        DOF : Global Degree of Freedom to take the derivitive w.r.t.

        Returns
        -------
        K : The dervivitve Stiffness matrix.

        '''
        if Mesh.ElementType == 'Q4':
            
            Element = dQ4dX
            
        if Mesh.ElementType == '5B':
            
            Element = d5BdX
        
        if Mesh.ElementType == 'Q8':
            
            Element = dQ8dX
            
        dRdX = np.zeros((Mesh.Nodes.shape[0]*2,1))
        
        # Find the node associated with the Global Degree of Freedom
        node = DOF//2
        
        # Find the elements that include the nodes, as well as the local node number
        element, LocalNodes = np.where(Mesh.Elements[:,1:] == node + 1)
        
        #Transform local node numbner to local degree of freedom
        if DOF%2:
            
            LocalDOF = LocalNodes*2 + 1
            
        else:
            
            LocalDOF = LocalNodes*2 
                
        for el, dof in zip(element, LocalDOF):
            
            #Node gloabl coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]
            
            #Node numbers in the element
            Local = Mesh.Elements[el,1:].reshape(-1,1)
            
            #coordinates in the gloabl stiff matrix
            RCoor = np.hstack((Local*2 - 2, Local*2 - 1))
            RCoor = RCoor.reshape(RCoor.size, )
            
            U_local = Mesh.U[RCoor,:]
            
            #Node gloabl coordinates
            NodeCoor = Mesh.Nodes[Mesh.Elements[el, 1:]-1, 1:]
            
            dRedX = Element(NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane,
                         LinearFlag = False, U = U_local).ResIntegrate(dof)
           
            dRdX[RCoor,:] += dRedX
              
        return dRdX
    
    def NonLinearSolver(Mesh, LoadSteps = 1, MaxIter = 8, tol = 1e-1, Sensitivity = False):
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
                Kff = Ktangent[np.ix_(Mesh.DegOfFreedom, Mesh.DegOfFreedom)]
                Rff = (GlobalResidual - LoadStep*(i+1))[Mesh.DegOfFreedom,:]
                
                Uff = np.linalg.solve(-1*Kff, Rff)
                
                Mesh.AllU[Mesh.DegOfFreedom, i + 1] += Uff[:,0]
                
                Mesh.U[Mesh.DegOfFreedom,:] += Uff
                
                if iter_cnt == 0:
                    
                    ResNorm0 = np.linalg.norm(Rff)
                
                ResNorm = np.linalg.norm(Rff)/ResNorm0
                
                print('Interation {}, Residual Norm {}'.format(iter_cnt, ResNorm))
                
                iter_cnt += 1
        
        if Sensitivity:
            
            Mesh.dUdx = np.zeros((Mesh.U.shape[0], Mesh.VariableNumber))
            
            for var in range(Mesh.VariableNumber):
            
                DOF = np.arange(0, Mesh.Nodes[-1,0].astype('int')*2, 1) # All Global degree of freedom vector
                dXdx = Mesh.dXdx[:, 2*var : 2*var+2].reshape(Mesh.U.shape[0]) # Derivitive of node coordinate w.r.t the design variable
                DOF = DOF[dXdx != 0] #remove zero terms
                dRdx = np.zeros_like(Mesh.U)
                
                for dof in DOF:
                    
                    dRdx += FEMSolvers._dRdX(Mesh, dof) * dXdx[dof]
      
                dUdX = np.linalg.solve(Kff, -1*dRdx[Mesh.DegOfFreedom,:] )
                
                Mesh.dUdx[Mesh.DegOfFreedom, var] += dUdX[:,0] 
            
        return 
    
    def ArcLengthSolver(Mesh, ArcLength, TotalArcLength, psi = 0, tol = 1e-4, MaxIter = 8, Sensitivity = False):
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

        Returns
        -------
        None.

        '''
        
        Mesh.U = np.zeros((Mesh.Nodes.shape[0]*2, 1)) # Initialize the displacement vector
        
        ResNorm = 1 #Intialize the norm of the residual vector
        
        Mesh.AllU = np.zeros((Mesh.Nodes.shape[0]*2, 1)) #store all displacment increments 

        SignDL = 1 #initialize sign direction for the load update
        Mesh.LoadValues = np.array([[0]]) # store load values
        Loadfactor = 0.0 #initialize the load factor value
        AccumulatedArcLength = 0 #initialize the Accumulated arc length

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
            ResNorm = 2*tol #Intialize the norm of the residual vector
            DU = np.zeros_like(Mesh.U[Mesh.DegOfFreedom,:]) #re-zero the update vector
            DL = 0 # re-zero the update to the load factor
            SignDL0 = SignDL # reset the sign direction
            
            while ResNorm > tol:
                
                Ktangent, GlobalResidual = FEMSolvers._ResAndTangentAssemble(Mesh)
                
                Kff = Ktangent[np.ix_(Mesh.DegOfFreedom, Mesh.DegOfFreedom)]
               
                Rff = (GlobalResidual - Mesh.Load*Loadfactor)[Mesh.DegOfFreedom,:]
                
                # Solve the two systems
                aQ = np.linalg.solve(Kff, Mesh.Load[Mesh.DegOfFreedom,:])
                aR = np.linalg.solve(-1*Kff, Rff)
                
                # Set up Constants for quadratic equation
                Uesti = DU + aR
                
                C1 = aQ.T.dot(aQ) + psi**2
                C2 = 2*(aQ.T.dot(Uesti)) + 2*psi**2*DL
                C3 = Uesti.T.dot(Uesti) + psi**2*DL**2 - ArcStep**2

                Discriminant = C2**2 - 4*C1*C3 #check if rational

                if Discriminant < 0 or iter_cnt > MaxIter : #if not, decrease Arc Length and zero values

                    print("------Adjusting Arc Length-------")

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
                # to do: fix dLdx
                dUdx = np.zeros((Mesh.U.shape[0], Mesh.VariableNumber))
                dLdx = np.zeros((1,Mesh.VariableNumber))
                
                
                dUdL = np.linalg.solve(Kff, -Loadfactor*Mesh.Load[Mesh.DegOfFreedom,:])
                
                dLdA = ArcStep/(Mesh.U[Mesh.DegOfFreedom,:].T @ dUdL + psi**2 * Loadfactor)
                dUdA = dUdL @ dLdA
               
                for var in range(Mesh.VariableNumber):
                
                    DOF = np.arange(0, Mesh.Nodes[-1,0].astype('int')*2, 1) # All Global degree of freedom vector
                    dXdx = Mesh.dXdx[:, 2*var : 2*var+2].reshape(Mesh.U.shape[0]) # Derivitive of node coordinate w.r.t the design variable
                    DOF = DOF[dXdx != 0] #remove zero terms
                    
                    dRdx = np.zeros_like(Mesh.U)

                    for dof in DOF:
                        
                        dRdx += FEMSolvers._dRdX(Mesh, dof) * dXdx[dof]
                    
                    B = np.vstack((-1*dRdx[Mesh.DegOfFreedom,:], 0))
    
                    Ka = np.hstack((Kff, np.zeros((len(Mesh.DegOfFreedom), 1))))
                    Ka = np.vstack((Ka, 
                         np.hstack((Mesh.U[Mesh.DegOfFreedom,:].T, Mesh.U[Mesh.DegOfFreedom,:].T @ dUdL))))
                    
                    ans = np.linalg.solve(Ka, B)

                    dUdx[Mesh.DegOfFreedom, var] += (dUdL*ans[-1,0] + ans[:-1,[0]])[:,0]
                    dLdx[0,var] = ans[-1,0]
                    
                if cnt_step == 0:
                    Mesh.dUdx_All = dUdx
                    Mesh.dLdx = np.copy(dLdx)

                else:
                    Mesh.dUdx_All = np.dstack((Mesh.dUdx_All, dUdx))
                    Mesh.dLdx = np.vstack((Mesh.dLdx, dLdx))

            cnt_step += 1
                  
        return 

            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
