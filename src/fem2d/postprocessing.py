from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.axes import Axes

from fem2d.elements.registry import LookupElement
from fem2d.materials import PlaneState

#: Components each field understands.
StressComponents = ('xx', 'yy', 'xy', 'vonMises')
StrainComponents = ('xx', 'yy', 'xy')

#: How values are taken to the nodes, where the field is discontinuous.
RecoveryMethods = ('extrapolate', 'average', 'none')


def _Element(Mesh, ElementNumber, LargeDeflection = False, U = None):
    '''
    Parameters
    ----------
    Mesh : Mesh object from the Meshers class.
    ElementNumber : Row index into Mesh.Elements.
    LargeDeflection : Build the element in its finite strain form.
    U : Full displacement vector to take the element's displacements from.
        Defaults to Mesh.U.

    Returns
    -------
    Element : Element object built on the requested element.
    DOF : Global degrees of freedom the element spans, in local order.

    '''
    ElementClass = LookupElement(Mesh.ElementType)

    Local = Mesh.Elements[ElementNumber, 1:]

    NodeCoor = Mesh.Nodes[Local - 1, 1:]

    DOF = np.vstack((Local*2 - 2, Local*2 - 1)).T.reshape(-1)

    if U is None:

        U = Mesh.U

    return ElementClass(NodeCoor, Mesh.t, Mesh.E, Mesh.v, Mesh.plane,
                        LinearFlag = not LargeDeflection,
                        U = np.asarray(U).reshape(-1, 1)[DOF]), DOF

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

    # Plane stress leaves the out of plane stress zero; plane strain does not.
    szz = 0.0 if PlaneState.From(Mesh.plane) is PlaneState.Stress else Mesh.v*(sxx + syy)

    return np.sqrt(0.5*((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6*sxy**2))


# ---------------------------------------------------------------- field recovery

def _AssumedStressBeta(Element, Ue):
    '''
    Parameters
    ----------
    Element : Element object.
    Ue : Element displacement vector.

    Returns
    -------
    The assumed stress parameters, or None if the element has no assumed stress
    field. Recovering them matters: for the five parameter element the
    displacement derived stress C @ B @ u is contaminated by the very locking
    the element exists to avoid, giving a shear of -3259 against the correct
    V/A = -50 on a cantilever, while P @ Beta gives -50 exactly.

    '''
    if not (hasattr(Element, 'P') and hasattr(Element, 'NumBeta')):

        return None

    gp, gw = Element.GaussPointsAndWeights()

    H = np.zeros((Element.NumBeta, Element.NumBeta))

    if Element.LinearFlag:

        G = np.zeros((Element.NumDOF, Element.NumBeta))

        for xi, wx in zip(gp, gw, strict = True):

            for eta, we in zip(gp, gw, strict = True):

                G += Element.Ge(xi, eta)*wx*we
                H += Element.He(xi, eta)*wx*we

        return np.linalg.solve(H, G.T @ Ue)

    # Finite strain: Beta follows from the strain rather than the displacement.
    M = np.zeros((Element.NumBeta, 1))

    for xi, wx in zip(gp, gw, strict = True):

        for eta, we in zip(gp, gw, strict = True):

            M += Element.Me(xi, eta)*wx*we
            H += Element.He(xi, eta)*wx*we

    return np.linalg.solve(H, M)


def _DeformationGradient(Element, xi, eta):
    '''
    Parameters
    ----------
    Element : Element object in its finite strain form.
    xi, eta : Local co-ordinates.

    Returns
    -------
    F as a 2x2 matrix. Fvec is stored as [F11, F22, F12, F21].

    '''
    f = Element.Fvec(xi, eta).ravel()

    return np.array([[f[0], f[2]],
                     [f[3], f[1]]])


def _StrainStressAt(Element, Ue, xi, eta, Beta = None):
    '''
    Parameters
    ----------
    Element : Element object, small strain or finite strain.
    Ue : Element displacement vector.
    xi, eta : Local co-ordinates.
    Beta : Assumed stress parameters, if the element has them.

    Returns
    -------
    Strain : [exx, eyy, gxy], engineering shear in both branches.
    Stress : [sxx, syy, sxy]. Cauchy stress in the finite strain branch.

    '''
    if Element.LinearFlag:

        Strain = (Element.B(xi, eta) @ Ue).ravel()

        if Beta is None:

            Stress = (Element.C() @ Element.B(xi, eta) @ Ue).ravel()

        else:

            Stress = (Element.P(xi, eta) @ Beta).ravel()

        return Strain, Stress[:3]

    # Green-Lagrange carries the tensor shear component twice, so it is doubled
    # here to report engineering shear as the small strain branch does.
    Evec = Element.Evec(xi, eta).ravel()

    Strain = np.array([Evec[0], Evec[1], 2*Evec[2]])

    Svec = Element.Svec(xi, eta).ravel() if Beta is None else (Element.P(xi, eta) @ Beta).ravel()

    # Push the 2nd Piola-Kirchhoff stress forward to Cauchy, which is the true
    # stress on the deformed body and what a von Mises value should mean.
    F = _DeformationGradient(Element, xi, eta)

    S = np.array([[Svec[0], Svec[2]],
                  [Svec[2], Svec[1]]])

    Sigma = F @ S @ F.T / np.linalg.det(F)

    return Strain, np.array([Sigma[0, 0], Sigma[1, 1], Sigma[0, 1]])


def _Component(Mesh, Strain, Stress, Kind, Component, DetF = None):
    '''
    Parameters
    ----------
    Mesh : Mesh object from the Meshers class.
    Strain, Stress : Arrays whose last axis is [xx, yy, xy].
    Kind : 'stress' or 'strain'.
    Component : Which component to pull out.
    DetF : det F per point, present only under finite strain.

    Returns
    -------
    The requested scalar field.

    '''
    Index = {'xx' : 0, 'yy' : 1, 'xy' : 2}

    if Component in Index:

        return (Stress if Kind == 'stress' else Strain)[..., Index[Component]]

    sxx, syy, sxy = Stress[..., 0], Stress[..., 1], Stress[..., 2]

    if PlaneState.From(Mesh.plane) is PlaneState.Stress:

        szz = np.zeros_like(sxx)

    elif DetF is None:

        # Plane strain, small strain: exact.
        szz = Mesh.v*(sxx + syy)

    else:

        # Plane strain, finite strain: E33 is zero, so the St Venant-Kirchhoff
        # material gives S33 = lambda*(E11 + E22), pushed forward by det F.
        Lambda = Mesh.E*Mesh.v/((1 + Mesh.v)*(1 - 2*Mesh.v))

        szz = Lambda*(Strain[..., 0] + Strain[..., 1])/DetF

    return np.sqrt(0.5*((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6*sxy**2))


def _SamplePoints(Mesh, ElementClass, Recovery):
    '''
    Parameters
    ----------
    Mesh : Mesh object from the Meshers class.
    ElementClass : The element class the mesh uses.
    Recovery : One of RecoveryMethods.

    Returns
    -------
    Points : Local co-ordinates to evaluate the field at.
    Fit : Matrix taking those samples to the element's nodes, or None when the
          samples already sit on the nodes.

    Notes
    -----
    'extrapolate' samples at the Gauss points, where the stress is most
    accurate, and fits back to the nodes with the pseudo inverse of N evaluated
    there. For Q4 at 2x2 that reproduces the classical extrapolation matrix,
    whose leading entry is 1 + sqrt(3)/2. The reference element does not depend
    on the geometry, so this is built once for the whole mesh.

    '''
    Probe = ElementClass(Mesh.Nodes[Mesh.Elements[0, 1:] - 1, 1:],
                         Mesh.t, Mesh.E, Mesh.v, Mesh.plane)

    if Recovery == 'extrapolate':

        gp, _ = Probe.GaussPointsAndWeights()

        Points = [(a, b) for a in gp for b in gp]

        A = np.array([Probe.ShapeFunctions(a, b) for a, b in Points])

        return Points, np.linalg.pinv(A)

    if ElementClass.LocalNodes is None:

        raise ValueError(
            f'{ElementClass.__name__} does not define LocalNodes, so the field cannot be '
            f'evaluated at its nodes. Either use Recovery=\'extrapolate\', which only ever '
            f'samples at the Gauss points, or give the class a LocalNodes array holding '
            f'each node\'s (xi, eta) co-ordinates.')

    return [tuple(p) for p in ElementClass.LocalNodes], None


def _ElementNodalFields(Mesh, U, LargeDeflection, Recovery):
    '''
    Parameters
    ----------
    Mesh : Mesh object from the Meshers class.
    U : Displacement vector to evaluate at.
    LargeDeflection : Use the finite strain measures.
    Recovery : One of RecoveryMethods.

    Returns
    -------
    Strain, Stress : (NumElements, NumNodes, 3) at each element's own nodes,
                     before anything is shared between elements.
    DetF : (NumElements, NumNodes), or None under small strain.

    '''
    ElementClass = LookupElement(Mesh.ElementType)

    Points, Fit = _SamplePoints(Mesh, ElementClass, Recovery)

    NumElements = Mesh.Elements.shape[0]
    NumNodes = ElementClass.NumNodes

    Strain = np.zeros((NumElements, NumNodes, 3))
    Stress = np.zeros((NumElements, NumNodes, 3))
    DetF = np.zeros((NumElements, NumNodes)) if LargeDeflection else None

    U = np.asarray(U).reshape(-1, 1)

    for el in range(NumElements):

        Element, DOF = _Element(Mesh, el, LargeDeflection = LargeDeflection, U = U)

        Ue = U[DOF]

        Beta = _AssumedStressBeta(Element, Ue)

        Sampled = np.array([_StrainStressAt(Element, Ue, xi, eta, Beta) for xi, eta in Points])

        if Fit is None:

            Strain[el], Stress[el] = Sampled[:, 0, :], Sampled[:, 1, :]

        else:

            Strain[el] = Fit @ Sampled[:, 0, :]
            Stress[el] = Fit @ Sampled[:, 1, :]

        if LargeDeflection:

            Jacobians = np.array([np.linalg.det(_DeformationGradient(Element, xi, eta))
                                  for xi, eta in Points])

            DetF[el] = Jacobians if Fit is None else Fit @ Jacobians

    return Strain, Stress, DetF


def _Share(Mesh, PerElement, Recovery, NumNodes):
    '''
    Parameters
    ----------
    Mesh : Mesh object from the Meshers class.
    PerElement : (NumElements, NumNodes, ...) values at each element's nodes.
    Recovery : One of RecoveryMethods.
    NumNodes : Nodes per element.

    Returns
    -------
    The values averaged onto the global nodes, or the per element values
    untouched when Recovery is 'none', in which case the caller duplicates the
    nodes so the jumps between elements stay visible.

    '''
    if Recovery == 'none':

        return PerElement

    Connect = Mesh.Elements[:, 1:1 + NumNodes] - 1

    Total = np.zeros((Mesh.Nodes.shape[0],) + PerElement.shape[2:])
    Count = np.zeros(Mesh.Nodes.shape[0])

    np.add.at(Total, Connect.ravel(), PerElement.reshape((-1,) + PerElement.shape[2:]))
    np.add.at(Count, Connect.ravel(), 1)

    return Total/Count.reshape((-1,) + (1,)*(Total.ndim - 1))



# ---------------------------------------------------------------- contour plots

@dataclass
class ContourOptions:
    '''
    Everything that controls how a contour is drawn.

    Attributes
    ----------
    Component : Which component to draw. None takes the field's own default,
                'vonMises' for stress and 'xx' for strain.
    Recovery : What happens at nodes and edges, where the field is
               discontinuous between elements.

               'extrapolate' samples at the Gauss points, where the stress is
               most accurate, fits those back to the element nodes and then
               averages. This is what commercial codes do and is the default.

               'average' evaluates at the nodes directly and averages there.
               Simpler, and the usual textbook description, but nodes are the
               least accurate place to sample a stress.

               'none' does not average at all. Each element keeps its own
               values so the jumps between elements stay visible, which is the
               most useful setting for judging whether a mesh is fine enough.
    step : Which column of Mesh.AllU to use, that is which load step or arc
           step. -1 is the last. A LinearSolver result has a single column, so
           the setting does nothing there. Independent of Deformed: the field
           always comes from this step, and Deformed only decides which
           co-ordinates it is drawn on.
    Deformed : Draw on the deformed geometry rather than the original.
    Scale : Magnification applied to the displacements when Deformed.
    Levels : Number of contour bands.
    Cmap : Matplotlib colour map.
    ShowMesh : Draw the element edges over the contour.
    ColourBar : Attach a colour bar.
    ax : Axes to draw on. One is created when this is None.

    '''

    Component: str | None = None

    Recovery: str = 'extrapolate'

    step: int = -1

    Deformed: bool = True

    Scale: float = 1.0

    Levels: int = 20

    Cmap: str = 'viridis'

    ShowMesh: bool = True

    ColourBar: bool = True

    ax: Axes | None = None


def _Coordinates(Mesh, Options):
    '''
    Parameters
    ----------
    Mesh : Mesh object from the Meshers class.
    Options : ContourOptions.

    Returns
    -------
    (NumNodes, 2) co-ordinates to draw on, deformed or not.

    '''
    Coordinates = Mesh.Nodes[:, [1, 2]]

    if not Options.Deformed:

        return Coordinates

    return Coordinates + Options.Scale*Mesh.AllU[:, [Options.step]].reshape(Coordinates.shape)


def _LocalTriangles(ElementClass):
    '''
    Parameters
    ----------
    ElementClass : The element class the mesh uses.

    Returns
    -------
    Triangles over the element's nodes, in local node numbering. Delaunay over
    LocalNodes handles any element, giving two triangles for Q4 and eight for
    Q8 with no per element type code. An element that does not define
    LocalNodes falls back to its first four nodes taken as the corners.

    '''
    if ElementClass.LocalNodes is None:

        return np.array([[0, 1, 2], [0, 2, 3]])

    Local = ElementClass.LocalNodes

    return mtri.Triangulation(Local[:, 0], Local[:, 1]).triangles


def NodalField(Mesh, Kind, Component = None, step = -1, Recovery = 'extrapolate'):
    '''
    Parameters
    ----------
    Mesh : Mesh object from the Meshers class, already solved.
    Kind : 'stress' or 'strain'.
    Component : Which component. None takes the field's default.
    step : Which column of Mesh.AllU to evaluate at.
    Recovery : One of RecoveryMethods.

    Returns
    -------
    Values : Nodal values. One per global node, or one per element per node
             when Recovery is 'none'.
    Coordinates : Matching (n, 2) co-ordinates on the undeformed mesh.
    Triangles : (m, 3) triangulation indexing into Values and Coordinates.

    '''
    if Kind not in ('stress', 'strain'):

        raise ValueError(f'Kind must be \'stress\' or \'strain\', not {Kind!r}')

    Allowed = StressComponents if Kind == 'stress' else StrainComponents

    if Component is None:

        Component = 'vonMises' if Kind == 'stress' else 'xx'

    if Component not in Allowed:

        raise ValueError(f'Component {Component!r} is not one of {Allowed} for {Kind}')

    if Recovery not in RecoveryMethods:

        raise ValueError(f'Recovery {Recovery!r} is not one of {RecoveryMethods}')

    Options = ContourOptions(Component = Component, Recovery = Recovery, step = step,
                             Deformed = False)

    return _FieldAndMesh(Mesh, Kind, Options)


def _FieldAndMesh(Mesh, Kind, Options):
    '''
    Parameters
    ----------
    Mesh : Mesh object from the Meshers class, already solved.
    Kind : 'stress' or 'strain'.
    Options : ContourOptions.

    Returns
    -------
    Values, Coordinates, Triangles ready to hand to a triangular contour.

    '''
    ElementClass = LookupElement(Mesh.ElementType)

    NumNodes = ElementClass.NumNodes

    LargeDeflection = getattr(Mesh, 'LargeDeflection', False)

    U = Mesh.AllU[:, Options.step]

    Strain, Stress, DetF = _ElementNodalFields(Mesh, U, LargeDeflection, Options.Recovery)

    Strain = _Share(Mesh, Strain, Options.Recovery, NumNodes)
    Stress = _Share(Mesh, Stress, Options.Recovery, NumNodes)

    if DetF is not None:

        DetF = _Share(Mesh, DetF, Options.Recovery, NumNodes)

    Values = _Component(Mesh, Strain, Stress, Kind, Options.Component, DetF)

    Coordinates = _Coordinates(Mesh, Options)

    Connect = Mesh.Elements[:, 1:1 + NumNodes] - 1

    LocalTriangles = _LocalTriangles(ElementClass)

    if Options.Recovery == 'none':

        # Duplicate the nodes so each element contours on its own, which is
        # what makes the jumps between elements visible.
        Coordinates = Coordinates[Connect].reshape(-1, 2)

        Offsets = (np.arange(Mesh.Elements.shape[0])*NumNodes)[:, None, None]

        Triangles = (LocalTriangles[None, :, :] + Offsets).reshape(-1, 3)

        Values = Values.reshape(-1)

    else:

        Triangles = Connect[:, LocalTriangles].reshape(-1, 3)

    return Values, Coordinates, Triangles


def _Contour(Mesh, Kind, Options = None):
    '''
    Parameters
    ----------
    Mesh : Mesh object from the Meshers class, already solved.
    Kind : 'stress' or 'strain'.
    Options : ContourOptions, or None for the defaults.

    Returns
    -------
    The axes the contour was drawn on.

    '''
    Options = ContourOptions() if Options is None else Options

    Allowed = StressComponents if Kind == 'stress' else StrainComponents

    Component = Options.Component

    if Component is None:

        Component = 'vonMises' if Kind == 'stress' else 'xx'

    if Component not in Allowed:

        raise ValueError(f'Component {Component!r} is not one of {Allowed} for {Kind}')

    if Options.Recovery not in RecoveryMethods:

        raise ValueError(f'Recovery {Options.Recovery!r} is not one of {RecoveryMethods}')

    Options = ContourOptions(**{**Options.__dict__, 'Component' : Component})

    Values, Coordinates, Triangles = _FieldAndMesh(Mesh, Kind, Options)

    ax = Options.ax

    if ax is None:

        fig = plt.figure()
        ax = fig.add_subplot(111)

    Triangulation = mtri.Triangulation(Coordinates[:, 0], Coordinates[:, 1], Triangles)

    Filled = ax.tricontourf(Triangulation, Values, levels = Options.Levels, cmap = Options.Cmap)

    if Options.ShowMesh:

        Outline = _Coordinates(Mesh, Options)

        NumNodes = LookupElement(Mesh.ElementType).NumNodes

        for Element in Mesh.Elements[:, 1:1 + NumNodes] - 1:

            Corners = Outline[np.append(Element[:4], Element[0])]

            ax.plot(Corners[:, 0], Corners[:, 1], 'k-', linewidth = 0.4, alpha = 0.5)

    if Options.ColourBar:

        ax.figure.colorbar(Filled, ax = ax, label = f'{Kind} {Component}')

    ax.axis('equal')

    return ax


class Plotting:

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
        if ax is None:

            fig = plt.figure()
            ax = fig.add_subplot(111)

        for cnt, pos in enumerate(Mesh.Elements[:,1:5].astype('int'), start = 1):

            pos = pos - 1

            ax.plot(Mesh.Nodes[pos,1], Mesh.Nodes[pos,2],'k.')

            for i in range(3):

                ax.plot([Mesh.Nodes[pos[i],1],Mesh.Nodes[pos[i+1],1]],[Mesh.Nodes[pos[i],2],Mesh.Nodes[pos[i+1],2]],'k')

            ax.plot([Mesh.Nodes[pos[-1],1],Mesh.Nodes[pos[0],1]],[Mesh.Nodes[pos[-1],2],Mesh.Nodes[pos[0],2]],'k')
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

        if ax is None:

            fig = plt.figure()
            ax = fig.add_subplot(111)

        Deformed = Mesh.Nodes[:,[1,2]] + Mesh.AllU[:, [step]].reshape(Mesh.Nodes[:,[1,2]].shape)
        for cnt, pos in enumerate(Mesh.Elements[:,1:5].astype('int'), start = 1):

            pos = pos - 1

            ax.plot(Deformed[pos,0], Deformed[pos,1],'k.')

            for i in range(3):

                ax.plot([Deformed[pos[i],0],Deformed[pos[i+1],0]],[Deformed[pos[i],1],Deformed[pos[i+1],1]],'k')

            ax.plot([Deformed[pos[3],0],Deformed[pos[0],0]],[Deformed[pos[-1],1],Deformed[pos[0],1]],'k')
            if shade:

                if cnt == Mesh.Elements[-1,0]:
                    ax.fill(Deformed[pos,0],Deformed[pos,1], alpha = alpha, color = c, label = 'Deformed')
                else:
                    ax.fill(Deformed[pos,0],Deformed[pos,1], alpha = alpha, color = c)

        ax.axis('equal')

        return


    def Overlay(Mesh, ax = None, alpha = 0.5, c = None, shade = True, steps = False):
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
        if c is None:

            c = ['b', 'r']

        if ax is None:

            fig = plt.figure()
            ax = fig.add_subplot(111)

        Plotting.InitialMesh(Mesh, ax = ax, c = c[0], alpha = 1, shade = True)

        if isinstance(steps, int) and not isinstance(steps, bool):

            LoadSteps = np.arange(0, Mesh.AllU.shape[1], steps)
            alphas = np.linspace(0.2, 1, len(LoadSteps))

        else:

            LoadSteps = [-1]
            alphas = [1]

        for s, a in zip(LoadSteps, alphas, strict=True):

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
        if ax is None:

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

    def StressContour(Mesh, Options = None):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the Meshers class, already solved.
        Options : ContourOptions. Component may be 'xx', 'yy', 'xy' or
                  'vonMises', defaulting to 'vonMises'.

        Returns
        -------
        The axes the contour was drawn on.

        '''
        return _Contour(Mesh, 'stress', Options)

    def StrainContour(Mesh, Options = None):
        '''
        Parameters
        ----------
        Mesh : Mesh object from the Meshers class, already solved.
        Options : ContourOptions. Component may be 'xx', 'yy' or 'xy',
                  defaulting to 'xx'.

        Returns
        -------
        The axes the contour was drawn on.

        '''
        return _Contour(Mesh, 'strain', Options)
