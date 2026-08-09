import matplotlib.pyplot as plt
import numpy as np
from fem2d.elements import FiveBeta, Q4, Q8

def _Element(Mesh, ElementNumber):
    '''
    Parameters
    ----------
    Mesh : Mesh object from the Meshers class.
    ElementNumber : Row index into Mesh.Elements.

    Returns
    -------
    Element : Element object built on the requested element.
    DOF : Global degrees of freedom the element spans, in local order.

    '''
    Types = {'Q4' : Q4, '5B' : FiveBeta, 'Q8' : Q8}

    if Mesh.ElementType not in Types:

        raise ValueError('Unknown ElementType {!r}, expected one of {}'.format(
                         Mesh.ElementType, sorted(Types)))

    Local = Mesh.Elements[ElementNumber, 1:]

    NodeCoor = Mesh.Nodes[Local - 1, 1:]

    DOF = np.vstack((Local*2 - 2, Local*2 - 1)).T.reshape(-1)

    return Types[Mesh.ElementType](NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane), DOF

def StrainMatrix(Mesh, ElementNumber, xi, eta):
    '''
    Parameters
    ----------
    Mesh : Mesh object from the Meshers class, already solved.
    ElementNumber : Row index into Mesh.Elements.
    xi : Local variable 1.
    eta : Local variable 2.

    Returns
    -------
    strain : Small strain vector [exx, eyy, gxy] at the local co-ordinates.

    '''
    Element, DOF = _Element(Mesh, ElementNumber)

    strain = Element.B(xi, eta) @ Mesh.U[DOF, :]

    return strain

def StressMatrix(Mesh, ElementNumber, xi, eta):
    '''
    Parameters
    ----------
    Mesh : Mesh object from the Meshers class, already solved.
    ElementNumber : Row index into Mesh.Elements.
    xi : Local variable 1.
    eta : Local variable 2.

    Returns
    -------
    stress : Small strain stress vector [sxx, syy, sxy] at the local co-ordinates.

    '''
    Element, DOF = _Element(Mesh, ElementNumber)

    stress = Element.C() @ Element.B(xi, eta) @ Mesh.U[DOF, :]

    return stress

def VonMises(Mesh, ElementNumber, xi, eta):
    '''
    Parameters
    ----------
    Mesh : Mesh object from the Meshers class, already solved.
    ElementNumber : Row index into Mesh.Elements.
    xi : Local variable 1.
    eta : Local variable 2.

    Returns
    -------
    The von Mises equivalent stress at the local co-ordinates.

    '''
    stress = StressMatrix(Mesh, ElementNumber, xi, eta)

    sxx, syy, sxy = stress[0,0], stress[1,0], stress[2,0]

    if Mesh.plane%2 == 0: #Plane Stress

        szz = 0.0

    else: #Plane Strain

        szz = Mesh.v*(sxx + syy)

    return np.sqrt(0.5*((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6*sxy**2))

class Plotting(object):

    def InitialMesh(Mesh, ax = None, alpha = 0.5, shade = True, c = 'b', label = 'Undeformed'):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the Meshers class.
        ax : ax object from matplotlib, 
            defualt is none where one is created.
        alpha : Darkness of the colour. The default is 0.5.
        shade : If the mesh is shaded.
        c : colour of the shading. The default is 'b'.
        label : label of the mesh if a legend is wanted.

        Returns
        -------
        A plot of the initial mesh.

        '''
        if ax == None:
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        cnt = 0
        for pos in Mesh.Elements[:,1:5].astype('int'):
            
            pos = pos - 1
            
            ax.plot(Mesh.Nodes[pos,1], Mesh.Nodes[pos,2],'k.')
            
            for i in range(3):
            
                ax.plot([Mesh.Nodes[pos[i],1],Mesh.Nodes[pos[i+1],1]],[Mesh.Nodes[pos[i],2],Mesh.Nodes[pos[i+1],2]],'k')
            
            ax.plot([Mesh.Nodes[pos[-1],1],Mesh.Nodes[pos[0],1]],[Mesh.Nodes[pos[-1],2],Mesh.Nodes[pos[0],2]],'k')
            cnt += 1
            if shade:
                
                if cnt == Mesh.Elements[-1,0]:
                    ax.fill(Mesh.Nodes[pos,1],Mesh.Nodes[pos,2], alpha = alpha, color = c, label = label)
                else:
                    ax.fill(Mesh.Nodes[pos,1],Mesh.Nodes[pos,2], alpha = alpha, color = c)  
        
        ax.axis('equal')
            
        return  ax
    
    def ProblemDiagram(Mesh, ax = None):
        # Not done
        ax = Plotting.InitialMesh(Mesh, ax = ax)
        
        #Number the Nodes
        for Node in Mesh.Nodes:
        
            ax.annotate('{}'.format(Node[0].astype('int')), Node[1:], fontsize = 8)
        
        #Number the elements
        for element in Mesh.Elements:
            
            posX = Mesh.Nodes[element[1:].astype('int') - 1,1].mean()
            posY = Mesh.Nodes[element[1:].astype('int') - 1,2].mean()
            
            ax.annotate('{}'.format(element[0].astype('int')), [posX, posY], fontsize = 10)
        
        return ax
        
    
    def DeformedMesh(Mesh, step = -1, ax = None, alpha = 0.5, shade = True, c = 'b', label = None):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the Meshers class.
        step: Which load step to be plotted. Only needed for Nonlinear Solver.
             Defaut is -1, for final load step.
        ax : ax object from matplotlib, 
            defualt is none where one is created.
        alpha : Darkness of the colour. The default is 0.5.
        shade : If the mesh is shaded.
        c : colour of the shading. The default is 'b'.
        label : label of the mesh if a legend is wanted.

        Returns
        -------
        A plot of the deformed mesh.

        '''
        
        if ax == None:
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        Deformed = Mesh.Nodes[:,[1,2]] + Mesh.AllU[:, [step]].reshape(Mesh.Nodes[:,[1,2]].shape)
        cnt = 0
        for pos in Mesh.Elements[:,1:5].astype('int'):
            
            pos = pos - 1
            
            ax.plot(Deformed[pos,0], Deformed[pos,1],'k.')
            
            for i in range(3):
            
                ax.plot([Deformed[pos[i],0],Deformed[pos[i+1],0]],[Deformed[pos[i],1],Deformed[pos[i+1],1]],'k')
            
            ax.plot([Deformed[pos[3],0],Deformed[pos[0],0]],[Deformed[pos[-1],1],Deformed[pos[0],1]],'k')
            cnt += 1
            if shade:
                
                if cnt == Mesh.Elements[-1,0]:
                    ax.fill(Deformed[pos,0],Deformed[pos,1], alpha = alpha, color = c, label = 'Deformed')
                else:
                    ax.fill(Deformed[pos,0],Deformed[pos,1], alpha = alpha, color = c)
                    
        ax.axis('equal')
                    
        return
        
    
    def Overlay(Mesh, ax = None, alpha = 0.5, c = ['b', 'r'], shade = True, steps = False):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the Meshers class.
        ax : ax object from matplotlib, 
            defualt is none where one is created.
        alpha : Darkness of the colour. The default is 0.5
        shade : If the mesh is shaded.
        c : colour of the shading. The default is 'b' and 'r'.
        label : label of the mesh if a legend is wanted.

        Returns
        -------
        An overlay plot of the initial and deformed mesh.

        '''
        if ax == None:
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        Plotting.InitialMesh(Mesh, ax = ax, c = c[0], alpha = 1, shade = True)
        
        if type(steps) == int:
            
            LoadSteps = np.arange(0, Mesh.AllU.shape[1], steps)
            alphas = np.linspace(0.2, 1, len(LoadSteps))
            
        else:
            
            LoadSteps = [-1]
            alphas = [1]
        
        for s, a in zip(LoadSteps, alphas):
        
            Plotting.DeformedMesh(Mesh, step = s, ax = ax, c = c[1], alpha = a, shade = True)
            
        return ax
    
    def LoadPath(Mesh, c = 'k', ax = None):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the Mesher Class.
        c : Colour of the line plot
            The default is 'k'.
        ax :  ax object from matplotlib, 
             defualt is none where one is created.

        Returns
        -------
        A plot of the load vs the displacement of the loaded node.

        '''
        if ax == None:
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set(xlabel = 'Displacement', ylabel = 'Load')
        
        if not hasattr(Mesh, 'LoadValues'):

            raise AttributeError(
                'This Mesh has no LoadValues, so there is no load path to plot. '
                'LinearSolver applies the load in one shot; use NonLinearSolver '
                'or ArcLengthSolver to trace a path.')

        Disp = Mesh.AllU[Mesh.LoadNode, :]
        Load = np.ravel(Mesh.LoadValues)*abs(Mesh.Load[Mesh.LoadNode, 0])

        ax.plot(Disp, Load, color = c, marker = '.')
        
            
        return ax
