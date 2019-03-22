"""Define zero dimensional geometric entities."""

import logging
from math import sqrt
from numbers import Number  # TODO: Use duck typing instead of type checking
import numpy as np

from xdesign.geometry.entity import *

logger = logging.getLogger(__name__)

__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'Point',
    'calc_standard',
    'dim',
    'rotated',
    'distance',
    'norm',
]


def dim(self):
    """Return the dimensionality of the ambient space."""
    return self.shape[-1]


def rotated(self, theta, center=None, axis=None):
    """Rotates theta radians around an axis."""
    if center is None:
        center = np.zeros(dim(self))
    if axis is not None:
        raise NotImplementedError(
            "Rotation about axis besides [0 0 1] are"
            " not implemented."
        )
    self = self.copy()
    # shift rotation center to origin
    self -= center
    # do rotation
    R = np.eye(dim(self))
    R[0, 0] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)
    R[1, 1] = np.cos(theta)
    self = np.dot(R, self)
    # shift rotation center back
    self += center
    return self


def distance(self, other):
    """Return the closest distance this and the other."""
    d = self - other
    return np.sqrt(d.dot(d))


def norm(self):
    """Euclidian (L2) norm of the vector."""
    # See http://stackoverflow.com/a/23576322 for a discussion of the
    # quickest way to calculate the norm of a vector.
    return np.sqrt(self.dot(self))


def calc_standard(A):
    """Return the standard equation for the row-wise points in A.

    The coefficents (c_{0}*x + ... = c_{1}) describe the
    hyper-plane defined by the row-wise N-dimensional points A.

    Parameters
    ----------
    A : :py:class:`np.array` (..., N, N)
        Each row is an N-dimensional point on the plane.

    Return
    ------
    c0 : :py:class:`np.array` (..., N)
        The first N coeffients for the hyper-plane
    c1 : :py:class:`np.array` (..., 1)
        The last coefficient for the hyper-plane

    """
    b = np.ones(A.shape[0:-1])
    x1 = np.atleast_1d(b[..., 0])
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        if str(e) != 'Singular matrix':
            raise
        else:
            a = A[..., 1, 1] - A[..., 0, 1]
            b = A[..., 0, 0] - A[..., 1, 0]
            x1 = np.atleast_1d(a * A[..., 0, 0] + b * A[..., 0, 1])
            x = np.stack([a, b], axis=-1)
    return x, x1


class Point(Entity):
    """Define a point in ND cartesian space.

    Parameters
    ----------
    x : :class:`.ndarray`, :class:`.list`
        ND coordinates of the point.

    Raises
    ------
    TypeError
        If x is not a list or ndarray.

    """

    def __init__(self, x):
        if not isinstance(x, (list, np.ndarray)):
            raise TypeError("x must be list, or array of coordinates.")

        super(Point, self).__init__()
        self._x = np.array(x, dtype=float, ndmin=1)
        self._x = np.ravel(self._x)
        self._dim = self._x.size

    def __repr__(self):
        """Return a string representation for easier debugging."""
        return "Point([%s" % ', '.join([repr(n) for n in self._x]) + "])"

    @property
    def x(self):
        """Dimension 0 of the point."""
        return self._x[0]

    @property
    def y(self):
        """Dimension 1 of the point."""
        return self._x[1]

    @property
    def z(self):
        """Dimension 2 of the point."""
        return self._x[2]

    @property
    def norm(self):
        """Euclidian (L2) norm of the vector to the point."""
        # See http://stackoverflow.com/a/23576322 for a discussion of the
        # quickest way to calculate the norm of a vector.
        return sqrt(self._x.dot(self._x))

    def translate(self, vector):
        """Translate the point along the given vector."""
        if not isinstance(vector, (list, np.ndarray)):
            raise TypeError("vector must be arraylike.")

        self._x += vector

    def rotate(self, theta, point=None, axis=None):
        """Rotates theta radians around an axis."""
        if not isinstance(theta, Number):
            raise TypeError("theta must be scalar.")
        if point is None:
            center = np.zeros(self.dim)
        elif isinstance(point, Point):
            center = point._x
        else:
            raise TypeError("center of rotation must be Point.")
        if axis is not None:
            raise NotImplementedError(
                "Rotation about axis besides [0 0 1] are"
                " not implemented."
            )

        # shift rotation center to origin
        self._x -= center

        # do rotation
        R = np.eye(self.dim)
        R[0, 0] = np.cos(theta)
        R[0, 1] = -np.sin(theta)
        R[1, 0] = np.sin(theta)
        R[1, 1] = np.cos(theta)

        self._x = np.dot(R, self._x)
        # shift rotation center back
        self._x += center

    def scale(self, vector):
        """SScale the ambient space in each dimension according to vector.

        Scaling is centered on the origin.
        """
        if not isinstance(vector, (List, np.ndarray)):
            raise TypeError("vector must be arraylike.")

        self._x *= vector

    def contains(self, other):
        """Return wether the other is within the bounds of the Point.

        Points can only contain other Points.
        """
        if isinstance(other, Point):
            return self == point
        if isinstance(other, np.ndarray):
            return np.all(self._x == other, axis=1)
        else:  # points can only contain points
            return False

    def collision(self, other):
        """Return True if this Point collides with another entity."""
        if isinstance(other, Point):
            return self == point
        else:
            raise NotImplementedError

    def distance(self, other):
        """Return the closest distance this and the other."""
        if not isinstance(other, Point):
            return other.distance(self)
        d = self._x - other._x
        return sqrt(d.dot(d))

    # OVERLOADS
    def __eq__(self, point):
        if not isinstance(point, Point):
            raise TypeError("Points can only equal to other points.")
        return np.array_equal(self._x, point._x)

    def __add__(self, point):
        """Addition."""
        if not isinstance(point, Point):
            raise TypeError("Points can only add to other points.")
        return Point(self._x + point._x)

    def __sub__(self, point):
        """Subtraction."""
        if not isinstance(point, Point):
            raise TypeError("Points can only subtract from other points.")
        return Point(self._x - point._x)

    def __mul__(self, c):
        """Scalar, vector multiplication."""
        if not isinstance(c, Number):
            raise TypeError("Points can only multiply scalars.")
        return Point(self._x * c)

    def __truediv__(self, c):
        """Scalar, vector division."""
        if not isinstance(c, Number):
            raise TypeError("Points can only divide scalars.")
        return Point(self._x / c)

    def __hash__(self):
        return hash(self._x[:])
