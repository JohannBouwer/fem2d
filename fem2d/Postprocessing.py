import matplotlib.pyplot as plt
import numpy as np
from Elements4Node import *

def StrainMatrix(Mesh, ElementNumber, GlobalCoor):
    
    NodeCoor = Mesh.Nodes[Mesh.Elements[ElementNumber, 1:] - 1, 1:]
    
    if Mesh.ElementTpye == 'q4':
        
        Element = Q4
    
    if Mesh.ElementType == '5B':
        
        Element = FiveBeta
        
    B = Element(NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane).B(xi, eta)
    
    strain = B.dot(U)
    
    return strain

def StressMatrix(Mesh, ElementNumber):

    NodeCoor = Mesh.Nodes[Mesh.Elements[ElementNumber, 1:] - 1, 1:]
    
    if Mesh.ElementTpye == 'q4':
        
        Element = Q4(NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane)
    
    if Mesh.ElementType == '5B':
        
        Element = FiveBeta(NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane)
    

    B = Element.B(xi, eta)
    C = Element.C()
    
    stress = C.dot(B).dot(U)
    
    return stress

def VonMisses(Mesh, ElementNumber, GlobalCoor):
    
    
    return

class Plotting(object):

    def InitialMesh(Mesh, ax = None, alpha = 0.5, shade = True, c = 'b', label = 'Undeformend'):
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
        
    
    def DeformendMesh(Mesh, step = -1, ax = None, alpha = 0.5, shade = True, c = 'b', label = None):
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
        A plot of the deformend mesh.

        '''
        
        if ax == None:
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        Deformend = Mesh.Nodes[:,[1,2]] + Mesh.AllU[:, [step]].reshape(Mesh.Nodes[:,[1,2]].shape)
        cnt = 0
        for pos in Mesh.Elements[:,1:5].astype('int'):
            
            pos = pos - 1
            
            ax.plot(Deformend[pos,0], Deformend[pos,1],'k.')
            
            for i in range(3):
            
                ax.plot([Deformend[pos[i],0],Deformend[pos[i+1],0]],[Deformend[pos[i],1],Deformend[pos[i+1],1]],'k')
            
            ax.plot([Deformend[pos[3],0],Deformend[pos[0],0]],[Deformend[pos[-1],1],Deformend[pos[0],1]],'k')
            cnt += 1
            if shade:
                
                if cnt == Mesh.Elements[-1,0]:
                    ax.fill(Deformend[pos,0],Deformend[pos,1], alpha = alpha, color = c, label = 'Deformend')
                else:
                    ax.fill(Deformend[pos,0],Deformend[pos,1], alpha = alpha, color = c)
                    
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
        An overlay plot of the initial and deformend mesh.

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
        
            Plotting.DeformendMesh(Mesh, step = s, ax = ax, c = c[1], alpha = a, shade = True)
            
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
        
        Disp = Mesh.AllU[Mesh.LoadNode, :]
        Load = Mesh.LoadValues*abs(Mesh.Load[Mesh.LoadNode, :])
        
        ax.plot(Disp, Load, color = c, marker = '.')
        
            
        return ax
