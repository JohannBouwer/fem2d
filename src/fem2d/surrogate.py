"""Surrogates of a solved load path, over the shape variables and the arc length together.

The thing being modelled is what the solver produced - a load factor, a displacement - and
not an objective computed from it. That is the standard way round, and the reason is that an
objective built on top of fitted responses can be changed, reweighted or replaced without
sampling anything again, whereas a surrogate of the objective has to be rebuilt from new
simulations every time the question changes.

Arc length is an input alongside the shape, so a stored path is not one sample but one sample
per arc step, each carrying a gradient in every input direction. The solver supplies the shape
derivatives; the arc-length derivatives are ArcLengthSolver's dLds and dUds_All.

ge_rbf handles trajectory data natively, so this module is now only the glue between the
solver's output and that API: stacking the paths into one block, keeping the trajectory labels,
and unwinding both coordinate maps on the way back out. Everything that used to be done by hand
here now has a name upstream -

    TrajectoryScaler          the arc-length stretch, and the gradient chain rule for it
    IsotropicTransformer      the frame, with fallback=True for the retreat
    RBFRegressor(centers=n)   space-filling centres drawn at fit time
    basis_search              the centre count and the shape parameter, chosen together
    scaled_epsilons           candidates relative to the sample spacing

Only importable with the optional ge_rbf dependency installed:

    uv sync --extra surrogate

Deliberately not imported by fem2d/__init__.py, so `import fem2d` keeps working without it.

    from fem2d.surrogate import PathSurrogate

References
----------
Bouwer, J., Wilke, D. N. and Kok, S. (2024). A novel and fully automated coordinate system
transformation scheme for near optimal surrogate construction. Computer Methods in Applied
Mechanics and Engineering, 419, 116648. doi:10.1016/j.cma.2023.116648
"""

import logging

import numpy as np

try:
    from ge_rbf import (
        IsotropicTransformer,
        RBFRegressor,
        TrajectoryScaler,
        basis_search,
        gradient_search,
        scaled_epsilons,
    )
except ImportError as Error:  # pragma: no cover - depends on how the package was installed
    raise ImportError(
        'fem2d.surrogate needs the optional ge_rbf dependency. Install it with '
        '`uv sync --extra surrogate`.'
    ) from Error

logger = logging.getLogger(__name__)

__all__ = ['PathSurrogate']


class PathSurrogate:
    '''
    One solved quantity as a smooth function of the shape variables and the arc length.

    Fit one of these per response. Two of them - a load factor and a displacement - describe a
    family of load paths completely enough to compute an objective from, and a third costs
    another fit rather than another simulation.

    Parameters
    ----------
    Centres : How many basis functions, and where.

        'search' (the default) chooses the count and the shape parameter together with
        ge_rbf's basis_search, holding out whole designs. That is the right default for this
        data and the reason is worth knowing: a gradient-enhanced model carries the sampled
        gradients as rows of its own fitting system, so it reproduces them better the more
        centres it is given, and a gradient-scored search can only ever pick the largest count
        on offer. Holding out whole trajectories is what makes the count choosable at all.

        None puts one centre at every sample, interpolating. On clustered path data that hands
        the basis the samples' own anisotropy and is measurably the worst option; it is here to
        be compared against.

        An integer fixes the count and searches only the shape parameter.
    Frame : Estimate an isotropic coordinate frame from the sampled gradients.
    Method : IsotropicTransformer method. Retreats through ge_rbf's FALLBACK_METHODS if it
        cannot produce a frame, which a load path with a limit point in it sometimes cannot.
    Stretch : Passed to TrajectoryScaler. After scaling, one arc step is this many times the
        distance between neighbouring designs.
    CentreCounts, Epsilons : Grids to search. None uses ge_rbf's defaults, which for the shape
        parameter means scaled_epsilons on the framed samples.
    Folds : Held-out folds for the centre-count search.
    MaxCondition, GradientWeight, Seed : Passed through.

    Attributes
    ----------
    Scaler_, Frame_ : The fitted trajectory scaling and coordinate frame.
    Search_ : The shape parameter search, a BasisSearchResult when Centres='search'.
    Model_ : The fitted RBFRegressor.
    Groups_ : The trajectory label of each sample row.
    '''

    def __init__(self, Centres='search', Frame: bool = True, Method: str = 'ge-lhm',
                 Stretch: float = 1.5, CentreCounts=None, Epsilons=None, Folds: int = 5,
                 MaxCondition: float = 1e13, GradientWeight='auto', Seed=0):

        self.Centres = Centres
        self.Frame = Frame
        self.Method = Method
        self.Stretch = Stretch
        self.CentreCounts = CentreCounts
        self.Epsilons = Epsilons
        self.Folds = Folds
        self.MaxCondition = MaxCondition
        self.GradientWeight = GradientWeight
        self.Seed = Seed

        return

    # ------------------------------------------------------------------ stacking

    @staticmethod
    def _Stack(X, S):
        '''Design variables and arc length as one block, arc length last.'''

        X = np.atleast_2d(np.asarray(X, dtype=float))
        S = np.ravel(np.asarray(S, dtype=float))

        if X.shape[0] == 1 and S.size > 1:
            X = np.repeat(X, S.size, axis=0)

        return np.column_stack((X, S))

    # ------------------------------------------------------------------ fitting

    def Fit(self, X, S, Y, dYdx, dYds, Groups=None) -> 'PathSurrogate':
        '''
        Fit one response over the shape and arc length space.

        One row per stored path point, so a single solve contributes as many rows as it took
        arc steps.

        Parameters
        ----------
        X : (n, NumVariables) design, repeated down each path.
        S : (n,) accumulated arc length at that point.
        Y : (n,) the response there.
        dYdx : (n, NumVariables) derivative with respect to the design variables.
        dYds : (n,) derivative with respect to arc length.
        Groups : (n,) which path each row came from. None infers it from the design columns,
            which is only safe when the design row repeats bit-exactly down a path - true when
            the block was built with np.repeat, and not when each row carries its own
            round-off. The labels are needed twice, by the scaler and by the fold split, so
            they are resolved once here and passed to both.

        Returns
        -------
        self
        '''

        Z = self._Stack(X, S)
        G = self._Stack(dYdx, dYds)
        Y = np.ravel(np.asarray(Y, dtype=float))

        if Z.shape[0] != Y.size:
            raise ValueError(f'{Z.shape[0]} sample rows but {Y.size} response values.')

        if Groups is None:
            _, Groups = np.unique(Z[:, :-1], axis=0, return_inverse=True)
        self.Groups_ = np.ravel(Groups)

        # Make the arc-length axis commensurate with the design spacing, or every local
        # curvature estimate is built from points strung along a single path.
        self.Scaler_ = TrajectoryScaler(stretch=self.Stretch).fit(Z, groups=self.Groups_)
        Zs = self.Scaler_.transform(Z)
        Gs = self.Scaler_.transform_gradient(G)

        if self.Frame:
            self.Frame_ = IsotropicTransformer(method=self.Method, fallback=True).fit(Zs, dy=Gs)
        else:
            self.Frame_ = IsotropicTransformer(method='identity').fit(Zs)

        Zt = self.Frame_.transform(Zs)
        Gt = self.Frame_.transform_gradient(Gs)

        Epsilons = scaled_epsilons(Zt) if self.Epsilons is None else self.Epsilons
        Template = RBFRegressor(gradient_weight=self.GradientWeight, random_state=self.Seed)

        if self.Centres == 'search':
            # Whole trajectories are held out, so there cannot be more folds than there are
            # of them. Four designs is a perfectly reasonable experiment here and would
            # otherwise fail on the default of five.
            Folds = min(self.Folds, self.Scaler_.n_trajectories_)
            if Folds < self.Folds:
                logger.info('only %d trajectories, so using %d folds rather than %d',
                            self.Scaler_.n_trajectories_, Folds, self.Folds)

            self.Search_ = basis_search(Template, Zt, Y, Gt,
                                        n_centers=self.CentreCounts, epsilons=Epsilons,
                                        score='kfold', k=Folds, groups=self.Groups_,
                                        max_condition=self.MaxCondition,
                                        random_state=self.Seed)
        else:
            Template.set_params(centers=self.Centres)
            self.Search_ = gradient_search(Template, Zt, Y, Gt, epsilons=Epsilons,
                                           max_condition=self.MaxCondition)

        self.Model_ = self.Search_.best_estimator
        self.Samples_ = Zt.shape[0]

        return self

    def Predict(self, X, S, Gradient: bool = False):
        '''
        The response at designs X and arc lengths S.

        Parameters
        ----------
        X : (n, NumVariables), or a single design broadcast against S.
        S : (n,) arc lengths.
        Gradient : Also return (dYdx, dYds), in design and arc-length units.

        Returns
        -------
        (n,) values, or (values, dYdx, dYds).
        '''

        Z = self._Stack(X, S)
        Value, Scaled = self.Model_.predict(
            self.Frame_.transform(self.Scaler_.transform(Z)), return_gradient=True)

        if not Gradient:
            return np.ravel(Value)

        # Back out through both maps, in the order they were applied. Dropping either leaves
        # the values right and the gradient in the wrong coordinates.
        G = self.Scaler_.inverse_transform_gradient(
            self.Frame_.inverse_transform_gradient(Scaled))

        return np.ravel(Value), G[:, :-1], G[:, -1]
