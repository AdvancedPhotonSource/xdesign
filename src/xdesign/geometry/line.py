"""Define one dimensional geometric entities."""

__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'Line',
    'Segment',
]

import logging
from math import sqrt
import numpy as np

from xdesign.geometry.entity import *
from xdesign.geometry.point import *

logger = logging.getLogger(__name__)


class LinearEntity(Entity):
    """Define a base class for linear entities.

    e.g. :class:`.Line`, :class:`.Segment`, and :class:`.Ray`.

    The constructor takes two unique :class:`.Point`.

    Attributes
    ----------
    p1 : Point
    p2 : Point

    """

    def __init__(self, p1, p2):
        if not isinstance(p1, Point) or not isinstance(p2, Point):
            raise TypeError("p1 and p2 must be Points")
        if p1 == p2:
            raise ValueError('Requires two unique Points.')
        if p1.dim != p2.dim:
            raise ValueError('Two Points must have same dimensionality.')
        self.p1 = p1
        self.p2 = p2
        self._dim = p1.dim

    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__, repr(self.p1), repr(self.p2)
        )

    @property
    def vertical(self):
        """Return True if line is vertical."""
        return self.p1.x == self.p2.x

    @property
    def horizontal(self):
        """Return True if line is horizontal."""
        return self.p1.y == self.p2.y

    @property
    def slope(self):
        """Return the slope of the line."""
        if self.vertical:
            return np.inf
        else:
            return ((self.p2.y - self.p1.y) / (self.p2.x - self.p1.x))

    @property
    def points(self):
        """Return the 2-tuple of points defining this linear entity."""
        return (self.p1, self.p2)

    @property
    def length(self):
        """Return the length of the segment between p1 and p2."""
        return self.p1.distance(self.p2)

    @property
    def tangent(self):
        """Return the unit tangent vector."""
        dx = (self.p2._x - self.p1._x) / self.length
        return Point(dx)

    @property
    def normal(self):
        """Return the unit normal vector."""
        dx = (self.p2._x - self.p1._x) / self.length
        R = np.array([[0, 1], [-1, 0]])
        n = np.dot(R, dx)
        return Point(n)

    @property
    def numpy(self):
        """Return row-size numpy array of p1 and p2."""
        return np.stack((self.p1._x, self.p2._x), axis=0)

    @property
    def list(self):
        """Return an list of coordinates where p1 is the first D coordinates
        and p2 is the next D coordinates."""
        return np.concatenate((self.p1._x, self.p2._x), axis=0)

    def translate(self, vector):
        """Translate the :class:`.LinearEntity` by the given vector."""
        self.p1.translate(vector)
        self.p2.translate(vector)

    def rotate(self, theta, point=None, axis=None):
        """Rotate the :class:`.LinearEntity` by theta radians around an axis
        defined by an axis and a point."""
        self.p1.rotate(theta, point, axis)
        self.p2.rotate(theta, point, axis)


class Line(LinearEntity):
    """Line in 2D cartesian space.

    The constructor takes two unique :class:`.Point`.

    Attributes
    ----------
    p1 : Point
    p2 : Point
    """

    def __init__(self, p1, p2):
        super(Line, self).__init__(p1, p2)

    def __str__(self):
        """Return line equation."""
        if self.vertical:
            return "x = %s" % self.p1.x
        elif self.dim == 2:
            return "y = %sx + %s" % (self.slope, self.yintercept)
        else:
            A, B = self.standard
            return "%sx " % '+ '.join([str(n) for n in A]) + "= " + str(B)

    def __eq__(self, line):
        return (self.slope, self.yintercept) == (line.slope, line.yintercept)

    def intercept(self, n):
        """Calculates the intercept for the nth dimension."""
        if n > self._dim:
            return 0
        else:
            A, B = self.standard
            if A[n] == 0:
                return np.inf
            else:
                return B / A[n]

    @property
    def xintercept(self):
        """Return the x-intercept."""
        if self.horizontal:
            return np.inf
        else:
            return self.p1.x - 1 / self.slope * self.p1.y

    @property
    def yintercept(self):
        """Return the y-intercept."""
        if self.vertical:
            return np.inf
        else:
            return self.p1.y - self.slope * self.p1.x

    @property
    def standard(self):
        """Returns coeffients for the first N-1 standard equation coefficients.
        The Nth is returned separately."""
        A = np.stack([self.p1._x, self.p2._x], axis=0)
        return calc_standard(A)

    def distance(self, other):
        """Returns the closest distance between entities."""
        # REF: http://geomalgorithms.com/a02-_lines.html
        if not isinstance(other, Point):
            raise NotImplementedError("Line to point distance only.")
        d = np.cross(self.tangent._x, other._x - self.p1._x)
        if self.dim > 2:
            return sqrt(d.dot(d))
        else:
            return abs(d)


class Ray(Line):
    """Ray in 2-D cartesian space.

    It is defined by two distinct points.

    Attributes
    ----------
    p1 : Point (source)
    p2 : Point (point direction)
    """

    def __init__(self, p1, p2):
        super(Ray, self).__init__(p1, p2)

    @property
    def source(self):
        """The point from which the ray emanates."""
        return self.p1

    @property
    def direction(self):
        """The direction in which the ray emanates."""
        return self.p2 - self.p1

    def distance(self, other):
        # REF: http://geomalgorithms.com/a02-_lines.html
        v = self.p2._x - self.p1._x
        w = other._x - self.p1._x

        c1 = np.dot(w, v)

        if c1 <= 0:
            return self.p1.distance(other)
        else:
            return super(Ray, self).distance(other)


class Segment(Line):
    """Defines a finite line segment from two unique points."""

    def __init__(self, p1, p2):
        super(Segment, self).__init__(p1, p2)

    @property
    def midpoint(self):
        """Return the midpoint of the line segment."""
        return Point.midpoint(self.p1, self.p2)

    def distance(self, other):
        """Return the distance to the other."""
        # REF: http://geomalgorithms.com/a02-_lines.html
        v = self.p2._x - self.p1._x
        w = other._x - self.p1._x

        c1 = np.dot(w, v)
        c2 = np.dot(v, v)

        if c1 <= 0:
            return self.p1.distance(other)
        elif c2 <= c1:
            return self.p2.distance(other)
        else:
            return super(Segment, self).distance(other)
