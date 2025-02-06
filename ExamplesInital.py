#%% Required Imports
import numpy as np
from Meshers import Mesh, Q8Mesh
from Solvers import FEMSolvers
from Postprocessing import Plotting

#%% Problem SetUp

E = 210000 #youngs modulous
v = 0.3 #possions ratio
t = 1 #thickness
plane = 1 #plane strain or plain stress
ELtype = '5B' #Element type

Beam = Mesh(E, v, t, plane, ELtype) #Create mesh object

Beam.SimpleBeam(20, 10, 0.5, 50) #Create a simple cantilever beam
#Beam.SingleElement(1, 1, 300)
#%% Sove the problem

FEMSolvers.LinearSolver(Beam)
ax = Plotting.Overlay(Beam)

#%% Sove the problem

FEMSolvers.NonLinearSolver(Beam, LoadSteps= 7, tol=0.99,Sensitivity = False) #Sove the problem
ax = Plotting.Overlay(Beam, steps = 2)
#%%

FEMSolvers.ArcLengthSolver(Beam, 5, 20, tol = 0.09, Sensitivity=False)
ax = Plotting.Overlay(Beam, steps = 1) #Overaly of inital and deformend meshes

#%%
#Beam.LeeFrame(40, [120, 120], -1)

var = [[100]]
for v in var:
    
    E = 72 #youngs modulous
    t = 5 #thickness
    plane = 2#plane strain or plain stress
    ELtype = 'Q4' #Element type

    Beam = Mesh(E, 0.3, t, plane, ELtype) #Create mesh object
    
    #Beam.SemiCircularArch(40, 100, v, 5, -1)
    #Beam.LeeFrame(40, [120, 120], -1)
    Beam.SemiCircularArch(30, 100, v, 5, -1)
    #Beam.SimpleBeam(20, 10, 0.5, 50)
    
    FEMSolvers.ArcLengthSolver(Beam, 100, 1200, tol = 0.1, Sensitivity=False)
    
    Plotting.Overlay(Beam, steps = 1)
    ax = Plotting.LoadPath(Beam)
#%%
FEMSolvers.NonLinearSolver(Beam)

Plotting.Overlay(Beam, steps = 1)
Plotting.LoadPath(Beam, ax = ax, c = 'r')

FEMSolvers.LinearSolver(Beam)

Plotting.Overlay(Beam)
