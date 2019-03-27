"""Define algorithms to support intersection calculation."""

__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'clip_SH',
    'halfspacecirc',
]

import logging
from math import asin, sqrt
import warnings

import numpy as np

from xdesign.geometry.point import *

logger = logging.getLogger(__name__)


def halfspacecirc(d, r):
    """Return the area of intersection between a circle and half-plane.

    Returns the smaller fraction of a circle split by a line d units away
    from the center of the circle.

    Parameters
    ----------
    d : scalar
        The distance from the line to the center of the circle
    r : scalar
        The radius of the circle

    Returns
    -------
    f : scalar
        The proportion of the circle in the smaller half-space

    Reference
    ---------
    Glassner, A. S. (Ed.). (2013). Graphics gems. Elsevier.

    """
    assert r > 0, "The radius must positive"
    assert d >= 0, "The distance must be positive or zero."

    if d >= r:  # The line is too far away to overlap!
        return 0

    f = 0.5 - d * sqrt(r**2 - d**2) / (np.pi * r**2) - asin(d / r) / np.pi

    # Returns the smaller fraction of the circle, so it can be at most 1/2.
    if f < 0 or 0.5 < f:
        if f < -1e-16:  # f will often be less than 0 due to rounding errors
            warnings.warn(
                "halfspacecirc was out of bounds, {}".format(f), RuntimeWarning
            )
        f = 0

    return f


def two_lines_intersect(l0A, l0b, l1A, l1b):
    A = np.stack([l0A, l1A], axis=0)
    b = np.stack([l0b, l1b], axis=0)
    x = np.linalg.solve(A, b)
    return x


def half_space(self, center):
    """Return the half space polytope respresentation of the Line."""
    A, B = self.standard
    # test for positive or negative side of line
    if not halfspace_has_point(A, B, center):
        A = -A
        B = -B
    return A, B


def halfspace_has_point(A, B, point):
    return np.dot(A, point._x) <= B


def clip_SH(clipEdges, polygon):
    """Clip a polygon using the Sutherland-Hodgeman algorithm.

    Parameters
    ----------
    clipEdges [[A, b], ...]
        half-spaces defined by coefficients

    polygon

    """
    outputList = polygon.vertices
    for clipEdge in clipEdges:
        # previous iteration output is this iteration input
        inputList = outputList
        outputList = list()
        if len(inputList) == 0:
            break
        S = inputList[-1]

        for E in inputList:

            if halfspace_has_point(clipEdge[0], clipEdge[1], E):

                if not halfspace_has_point(clipEdge[0], clipEdge[1], S):
                    A, b = calc_standard(np.stack([S._x, E._x], axis=0))
                    new_vert = two_lines_intersect(
                        A, b, clipEdge[0], clipEdge[1]
                    )
                    outputList.append(Point(new_vert))

                outputList.append(E)

            elif halfspace_has_point(clipEdge[0], clipEdge[1], S):
                A, b = calc_standard(np.stack([S._x, E._x], axis=0))
                new_vert = two_lines_intersect(A, b, clipEdge[0], clipEdge[1])
                outputList.append(Point(new_vert))

            S = E
    return outputList
