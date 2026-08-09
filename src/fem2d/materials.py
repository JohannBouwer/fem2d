"""Plane state and the constitutive relationship."""

from enum import IntEnum

import numpy as np


class PlaneState(IntEnum):
    '''
    Which two dimensional idealisation the constitutive matrix uses.

    The values keep the original even/odd convention working, so an integer
    passed to a mesher or an element still selects the same state it always
    did: even is plane stress, odd is plane strain.
    '''

    Stress = 0

    Strain = 1

    @classmethod
    def From(cls, plane):
        '''
        Parameters
        ----------
        plane : A PlaneState, or any integer under the even/odd convention.

        Returns
        -------
        The matching PlaneState.
        '''

        if isinstance(plane, cls):

            return plane

        return cls(int(plane) % 2)


def ConstitutiveMatrix(E, v, plane, LinearFlag = True):
    '''
    Parameters
    ----------
    E : Youngs Modulus.
    v : Poissons ratio.
    plane : PlaneState, or an integer under the even/odd convention.
    LinearFlag : Small strain gives the 3x3 engineering shear form; the
                 nonlinear form is 4x4 because the shear term is carried twice,
                 once for each off diagonal of the symmetric tensor.

    Returns
    -------
    Cmat : Constitutive relationship matrix. i.e, stress-strain relationship.
    '''

    plane = PlaneState.From(plane)

    if LinearFlag:

        if plane is PlaneState.Stress:

            Cmat = E/(1 - v**2) * np.array([[1, v, 0],
                                            [v, 1, 0],
                                            [0, 0, (1 - v)/2]])
        else: #Plane Strain

            Cmat = E/((1 + v)*(1 - 2*v)) * np.array([[1 - v, v, 0],
                                                     [v, 1 - v, 0],
                                                     [0, 0, (1 - 2*v)/2]])

    else:

        if plane is PlaneState.Stress:

            Cmat = E/(1 - v**2) * np.array([[1, v, 0, 0],
                                            [v, 1, 0, 0],
                                            [0, 0, 1 - v, 0],
                                            [0, 0, 0, 1 - v]])

        else: #Plane strain via the standard substitution on E and v

            E = E/(1 - v**2)
            v = v/(1 - v)

            Cmat = E/(1 - v**2) * np.array([[1, v, 0, 0],
                                            [v, 1, 0, 0],
                                            [0, 0, 1 - v, 0],
                                            [0, 0, 0, 1 - v]])

    return Cmat
