#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2016, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2016. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################
"""Defines geometric objects to support :class:`.Phantom` definition and
perform compuational geometry for :mod:`.acquisition`.

.. moduleauthor:: Doga Gursoy <dgursoy@aps.anl.gov>
.. moduleauthor:: Daniel J Ching <carterbox@users.noreply.github.com>
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
import warnings
import matplotlib.pyplot as plt
from matplotlib.path import Path
from numbers import Number
import polytope as pt
from cached_property import cached_property
import copy
from math import sqrt, asin
from copy import deepcopy

logger = logging.getLogger(__name__)

__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Entity',
           'Point',
           'Circle',
           'Line',
           'Polygon',
           'Triangle',
           'Rectangle',
           'Square',
           'Mesh']


class Entity(object):
    """Base class for all geometric entities. All geometric entities should
    have these attributes and methods.

    Example
    -------
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_numpy.py


    Section breaks are created with two blank lines. Section breaks are also
    implicitly created anytime a new section starts. Section bodies *may* be
    indented:

    Parameters
    ----------
    x : :class:`.ndarray`, :class:`.list`
        ND coordinates of the point.

    Notes
    -----
        This is an example of an indented section. It's like any other section,
        but the body is indented to help it stand out from surrounding text.

    If a section is indented, then a section break is created by
    resuming unindented text.


    .. note::
        There are many other directives such as versionadded, versionchanged,
        rubric, centered, ... See the sphinx documentation for more details.

    """

    def __init__(self):
        self._dim = 0

    def __repr__(self):
        """A string representation for easier debugging.

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.
        """
        raise NotImplementedError

    @property
    def dim(self):
        """The dimensionality of the points which describe the entity.
        """
        return self._dim

    def translate(self, vector):
        """Translates the entity in the direction of a vector.

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.
        """
        raise NotImplementedError

    def rotate(self, theta, point=None, axis=None):
        """Rotates the entity theta radians around an axis defined by a point
        and a vector

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.
        """
        raise NotImplementedError

    def scale(self, vector):
        """Scales the entity in each dimension according to vector. Scaling is
        centered on the origin.

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.
        """
        raise NotImplementedError

    def contains(self, other):
        """Return whether this Entity strictly contains the other entity.

        Points on edges are contained by the Entity.

        Returns a boolean for all :class:`Entitity`. Returns an array of
        boolean for MxN size arrays where M is the number of points and N is
        the dimensionality.

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.
        """
        raise NotImplementedError

    def collision(self, other):
        """Returns True if this entity collides with another entity.

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.
        """
        raise NotImplementedError

    def distance(self, other):
        """Returns the closest distance between entities.

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.
        """
        raise NotImplementedError

    def midpoint(self, other):
        """Returns the midpoint between entities."""
        return self.distance(other) / 2.


class Point(Entity):
    """A point in ND cartesian space.

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
        """Calculates the euclidian (L2) norm of the vector to the point."""
        # See http://stackoverflow.com/a/23576322 for a discussion of the
        # quickest way to calculate the norm of a vector.
        return sqrt(self._x.dot(self._x))

    def translate(self, vector):
        """Translates the point along the given vector."""
        if not isinstance(vector, (list, np.ndarray)):
            raise TypeError("vector must be arraylike.")

        self._x += vector

    def rotate(self, theta, point=None, axis=None):
        """Rotates the point theta radians around the axis defined by the given
        point and axis."""
        if not isinstance(theta, Number):
            raise TypeError("theta must be scalar.")
        if point is None:
            center = np.zeros(self.dim)
        elif isinstance(point, Point):
            center = point._x
        else:
            raise TypeError("center of rotation must be Point.")
        if axis is not None:
            raise NotImplementedError("Rotation about axis besides [0 0 1] are"
                                      " not implemented.")

        # shift rotation center to origin
        self._x -= center
        # do rotation
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        self._x = np.dot(R, self._x)
        # shift rotation center back
        self._x += center

    def scale(self, vector):
        """Scales the coordinates of the point in each dimension according to
        the given vector. Scaling is centered on the origin."""
        if not isinstance(vector, (List, np.ndarray)):
            raise TypeError("vector must be arraylike.")

        self._x *= vector

    def contains(self, other):
        """Return wether the other is within the bounds of the Point. Points
        can only contain other Points."""
        if isinstance(other, Point):
            return self == point
        if isinstance(other, np.ndarray):
            return np.all(self._x == other, axis=1)
        else:  # points can only contain points
            return False

    def collision(self, other):
        """Returns True if this Point collides with another entity."""
        if isinstance(other, Point):
            return self == point
        else:
            raise NotImplementedError

    def distance(self, other):
        """Returns the closest distance between entities."""
        if isinstance(other, LinearEntity):
            return other.distance(self)
        elif not isinstance(other, Point):
            raise NotImplementedError("Point to point distance only.")
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


class LinearEntity(Entity):
    """Base class for linear entities in 2D Cartesian space. e.g. :class:`.Line`,
    :class:`.Segment`, and :class:`.Ray`.

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
        return "{}({}, {})".format(type(self).__name__, repr(self.p1),
                                   repr(self.p2))

    @property
    def vertical(self):
        """True if line is vertical."""
        return self.p1.x == self.p2.x

    @property
    def horizontal(self):
        """True if line is horizontal."""
        return self.p1.y == self.p2.y

    @property
    def slope(self):
        """Returns the slope of the line."""
        if self.vertical:
            return np.inf
        else:
            return ((self.p2.y - self.p1.y) /
                    (self.p2.x - self.p1.x))

    @property
    def points(self):
        """Returns the two points used to define this linear entity as a
        2-tuple."""
        return (self.p1, self.p2)

    # @property
    # def bounds(self):
    #     """Return a tuple (xmin, ymin, xmax, ymax) representing the
    #     bounding rectangle for the geometric figure.
    #     """
    #     xs = [p.x for p in self.points]
    #     ys = [p.y for p in self.points]
    #     return (min(xs), min(ys), max(xs), max(ys))

    @property
    def length(self):
        """Returns the length of the segment between p1 and p2."""
        return self.p1.distance(self.p2)

    @property
    def tangent(self):
        """Returns the unit tangent vector."""
        dx = (self.p2._x - self.p1._x) / self.length
        return Point(dx)

    @property
    def normal(self):
        """Return the unit normal vector."""
        dx = (self.p2._x - self.p1._x) / self.length
        R = np.array([[0, 1],
                      [-1,  0]])
        n = np.dot(R, dx)
        return Point(n)

    @property
    def numpy(self):
        """Returns an array of coordinates where the first row is p1 and the
        second row is p2."""
        return np.stack((self.p1._x, self.p2._x), axis=0)

    @property
    def list(self):
        """Returns an list of coordinates where p1 is the first D coordinates
        and p2 is the next D coordinates."""
        return np.concatenate((self.p1._x, self.p2._x), axis=0)

    def translate(self, vector):
        """Translates the :class:`.LinearEntity` by the given vector."""
        self.p1.translate(vector)
        self.p2.translate(vector)

    def rotate(self, theta, point=None, axis=None):
        """Rotates the :class:`.LinearEntity` by theta radians around an axis
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
                return B/A[n]

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
    """Segment in 2-D cartesian space.

    It is defined by two distinct points.

    Attributes
    ----------
    p1 : Point
    p2 : Point
    """

    def __init__(self, p1, p2):
        super(Segment, self).__init__(p1, p2)

    @property
    def midpoint(self):
        """The midpoint of the line segment."""
        return Point.midpoint(self.p1, self.p2)

    def distance(self, other):
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


class Curve(Entity):
    """The base class for closed manifolds defined by a single equation. e.g.
    :class:`.Circle`, :class:`.Sphere`, or :class:`.Torus`.

    Attributes
    ----------
    center : Point
    """
    def __init__(self, center):
        if not isinstance(center, Point):
            raise TypeError("center must be a Point.")
        super(Curve, self).__init__()
        self.center = center

    def __repr__(self):
        return "{}(center={})".format(type(self).__name__, repr(self.center))

    def translate(self, vector):
        """Translates the Curve along a vector."""
        if not isinstance(vector, (Point, list, np.array)):
            raise TypeError("vector must be point, list, or array.")
        self.center.translate(vector)

    def rotate(self, theta, point=None, axis=None):
        """Rotates the Curve by theta radians around an axis which passes
        through a point radians."""
        self.center.rotate(theta, point, axis)


class Superellipse(Curve):
    """A Superellipse in 2D cartesian space.

    Attributes
    ----------
    center : Point
    a : scalar
    b : scalar
    n : scalar
    """

    def __init__(self, center, a, b, n):
        super(Superellipse, self).__init__(center)
        self.a = float(a)
        self.b = float(b)
        self.n = float(n)

    def __repr__(self):
        return "Superellipse(center={}, a={}, b={}, n={})".format(repr(self.center),
                                                                  repr(self.a),
                                                                  repr(self.b),
                                                                  repr(self.n))

    @property
    def list(self):
        """Return list representation."""
        return [self.center.x, self.center.y, self.a, self.b, self.n]

    def scale(self, val):
        """Scale."""
        self.a *= val
        self.b *= val


class Ellipse(Superellipse):
    """Ellipse in 2-D cartesian space.

    Attributes
    ----------
    center : Point
    a : scalar
    b : scalar
    """

    def __init__(self, center, a, b):
        super(Ellipse, self).__init__(center, a, b, 2)

    def __repr__(self):
        return "Ellipse(center={}, a={}, b={})".format(repr(self.center),
                                                       repr(self.a),
                                                       repr(self.b))

    @property
    def list(self):
        """Return list representation."""
        return [self.center.x, self.center.y, self.a, self.b]

    @property
    def area(self):
        """Return area."""
        return np.pi * self.a * self.b

    def scale(self, val):
        """Scale."""
        self.a *= val
        self.b *= val


class Circle(Curve):
    """Circle in 2D cartesian space.

    Attributes
    ----------
    center : Point
        The center point of the circle.
    radius : scalar
        The radius of the circle.
    sign : int (-1 or 1)
        The sign of the area
    """

    def __init__(self, center, radius, sign=1):
        super(Circle, self).__init__(center)
        self.radius = float(radius)
        self.sign = sign

    def __repr__(self):
        return "Circle(center={}, radius={}, sign={})".format(
            repr(self.center), repr(self.radius), repr(self.sign))

    def __str__(self):
        """Return the analytical equation."""
        return "(x-%s)^2 + (y-%s)^2 = %s^2" % (self.center.x, self.center.y,
                                               self.radius)

    def __eq__(self, circle):
        return ((self.x, self.y, self.radius) ==
                (circle.x, circle.y, circle.radius))

    def __neg__(self):
        copE = deepcopy(self)
        copE.sign = -copE.sign
        return copE

    @property
    def list(self):
        """Return list representation for saving to files."""
        return [self.center.x, self.center.y, self.radius]

    @property
    def circumference(self):
        """Returns the circumference."""
        return 2 * np.pi * self.radius

    @property
    def diameter(self):
        """Returns the diameter."""
        return 2 * self.radius

    @property
    def area(self):
        """Return the area."""
        return self.sign * np.pi * self.radius**2

    @property
    def patch(self):
        """Returns a matplotlib patch."""
        return plt.Circle((self.center.x, self.center.y), self.radius)

    # def scale(self, val):
    #     """Scale."""
    #     raise NotImplementedError
    #     self.center.scale(val)
    #     self.rad *= val

    def contains(self, other):
        """Return whether `other` is a proper subset.

        Return one boolean for all geometric entities. Return an array of
        boolean for array input.
        """
        if isinstance(other, Point):
            x = other._x
        elif isinstance(other, np.ndarray):
            x = other
        elif isinstance(other, Mesh):
            for face in other.faces:
                if not self.contains(face) and face.sign is 1:
                    return False
            return True
        else:
            if self.sign is 1:
                if other.sign is -1:
                    # Closed shape cannot contain infinite one
                    return False
                else:
                    assert other.sign is 1
                    # other is within A
                    if isinstance(other, Circle):
                        return (other.center.distance(self.center)
                                + other.radius < self.radius)
                    elif isinstance(other, Polygon):
                        x = _points_to_array(other.vertices)
                        return np.all(self.contains(x))

            elif self.sign is -1:
                if other.sign is 1:
                    # other is outside A and not around
                    if isinstance(other, Circle):
                        return (other.center.distance(self.center)
                                - other.radius > self.radius)
                    elif isinstance(other, Polygon):
                        x = _points_to_array(other.vertices)
                        return (np.all(self.contains(x)) and
                                not other.contains(-self))

                else:
                    assert other.sign is -1
                    # other is around A
                    if isinstance(other, Circle):
                        return (other.center.distance(self.center)
                                + self.radius < other.radius)
                    elif isinstance(other, Polygon):
                        return (-other).contains(-self)

        x = np.atleast_2d(x)

        if self.sign is 1:
            return np.sum((x - self.center._x)**2, axis=1) < self.radius**2
        else:
            return np.sum((x - self.center._x)**2, axis=1) > self.radius**2


def _points_to_array(points):
    a = np.zeros((len(points), points[0].dim))

    for i in range(len(points)):
        a[i] = points[i]._x

    return np.atleast_2d(a)


class Polygon(Entity):
    """A convex polygon in 2D cartesian space.

    It is defined by a number of distinct vertices of class :class:`.Point`.
    Superclasses include :class:`.Square`, :class:`.Triangle`, etc.

    Attributes
    ----------
    vertices : List of Points
    sign : int (-1 or 1)
        The sign of the area
    """

    def __init__(self, vertices, sign=1):
        for v in vertices:
            if not isinstance(v, Point):
                raise TypeError("vertices must be of type Point.")
        super(Polygon, self).__init__()
        self.vertices = vertices
        self._dim = vertices[0].dim
        self.sign = sign

    def __repr__(self):
        return "{}(vertices={}, sign={})".format(type(self).__name__,
                                                 repr(self.vertices),
                                                 repr(self.sign))

    def __str__(self):
        return "{}({})".format(type(self).__name__, str(self.numpy))
    # return "Polygon(%s" % ', '.join([str(n) for n in self.vertices]) + ")"

    def __neg__(self):
        copE = deepcopy(self)
        copE.sign = -copE.sign
        return copE

    @property
    def numverts(self):
        return len(self.vertices)

    @property
    def list(self):
        """Return list representation."""
        lst = []
        for m in range(self.numverts):
            lst.append(self.vertices[m].list)
        return lst

    @property
    def numpy(self):
        """Return Numpy representation."""
        return _points_to_array(self.vertices)

    @property
    def area(self):
        """Returns the area of the Polygon."""
        raise NotImplementedError

    @property
    def perimeter(self):
        """Return the perimeter of the Polygon."""
        perimeter = 0
        verts = self.vertices
        points = verts + [verts[0]]
        for m in range(self.numverts):
            perimeter += points[m].distance(points[m + 1])
        return perimeter

    @property
    def center(self):
        """The center of the bounding circle."""
        c = Point(np.zeros(self._dim))
        for m in range(self.numverts):
            c = c + self.vertices[m]
        return c / self.numverts

    @property
    def radius(self):
        """The radius of the bounding circle."""
        r = 0
        c = self.center
        for m in range(self.numverts):
            r = max(r, self.vertices[m].distance(c))
        return r

    @property
    def bounds(self):
        """Returns a 4-tuple (xmin, ymin, xmax, ymax) representing the
        bounding rectangle for the Polygon.
        """
        xs = [p.x for p in self.vertices]
        ys = [p.y for p in self.vertices]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def patch(self):
        """Returns a matplotlib patch."""
        return plt.Polygon(self.numpy)

    def translate(self, vector):
        """Translates the polygon by a vector."""
        for v in self.vertices:
            v.translate(vector)

        if 'half_space' in self.__dict__:
            self.half_space = self.half_space.translation(vector)

    def rotate(self, theta, point=None, axis=None):
        """Rotates the Polygon around an axis which passes through a point by
        theta radians."""
        for v in self.vertices:
            v.rotate(theta, point, axis)

        if 'half_space' in self.__dict__:
            if point is None:
                d = 0
            else:
                d = point._x
            self.half_space = self.half_space.translation(-d)
            self.half_space = self.half_space.rotation(0, 1, theta)
            self.half_space = self.half_space.translation(d)

    @property
    def edges(self):
        """Return a list of lines connecting the points of the Polygon."""
        edges = []

        for i in range(self.numverts):
            edges.append(Segment(self.vertices[i],
                                 self.vertices[(i+1) % self.numverts]))

        return edges

    def contains(self, other):
        """Return whether this Polygon contains the other."""

        if isinstance(other, Point):
            x = other._x
        elif isinstance(other, np.ndarray):
            x = other
        elif isinstance(other, Mesh):
            for face in other.faces:
                if not self.contains(face) and face.sign is 1:
                    return False
            return True
        else:
            if self.sign is 1:
                if other.sign is -1:
                    # Closed shape cannot contain infinite one
                    return False
                else:
                    assert other.sign is 1
                    # other is within A
                    if isinstance(other, Circle):
                        if self.contains(other.center):
                            for edge in self.edges:
                                if other.center.distance(edge) < other.radius:
                                    return False
                            return True
                        return False
                    elif isinstance(other, Polygon):
                        x = _points_to_array(other.vertices)
                        return np.all(self.contains(x))

            elif self.sign is -1:
                if other.sign is 1:
                    # other is outside A and not around
                    if isinstance(other, Circle):
                        if self.contains(other.center):
                            for edge in self.edges:
                                if other.center.distance(edge) < other.radius:
                                    return False
                            return True and not other.contains(-self)
                        return False
                    elif isinstance(other, Polygon):
                        x = _points_to_array(other.vertices)
                        return (np.all(self.contains(x)) and
                                not other.contains(-self))

                else:
                    assert other.sign is -1
                    # other is around A
                    if isinstance(other, Circle) or isinstance(other, Polygon):
                        return (-other).contains(-self)

        border = Path(self.numpy)

        if self.sign is 1:
            return border.contains_points(np.atleast_2d(x))
        else:
            return np.logical_not(border.contains_points(np.atleast_2d(x)))

    @cached_property
    def half_space(self):
        """Returns the half space polytope respresentation of the polygon."""
        assert(self.dim > 0), self.dim
        A = np.ndarray((self.numverts, self.dim))
        B = np.ndarray(self.numverts)

        for i in range(0, self.numverts):
            edge = Line(self.vertices[i], self.vertices[(i+1) % self.numverts])
            A[i, :], B[i] = edge.standard

            # test for positive or negative side of line
            if self.center._x.dot(A[i, :]) > B[i]:
                A[i, :] = -A[i, :]
                B[i] = -B[i]

        p = pt.Polytope(A, B)
        return p


class Triangle(Polygon):
    """Triangle in 2D cartesian space.

    It is defined by three distinct points.
    """

    def __init__(self, p1, p2, p3):
        super(Triangle, self).__init__([p1, p2, p3])

    def __repr__(self):
        return "Triangle({}, {}, {})".format(self.vertices[0],
                                             self.vertices[1],
                                             self.vertices[2])

    @property
    def center(self):
        center = Point([0, 0])
        for v in self.vertices:
            center += v
        return center / 3

    @property
    def area(self):
        A = self.vertices[0] - self.vertices[1]
        B = self.vertices[0] - self.vertices[2]
        return self.sign * 1/2 * np.abs(np.cross([A.x, A.y], [B.x, B.y]))


class Rectangle(Polygon):
    """Rectangle in 2D cartesian space.

    It is defined by four distinct points.
    """

    def __init__(self, p1, p2, p3, p4):
        super(Rectangle, self).__init__([p1, p2, p3, p4])

    def __repr__(self):
        return "Rectangle({}, {}, {}, {})".format(self.vertices[0],
                                                  self.vertices[1],
                                                  self.vertices[2],
                                                  self.vertices[3])

    @property
    def center(self):
        center = Point([0, 0])
        for v in self.vertices:
            center += v
        return center / 4

    @property
    def radius(self):
        """The radius of the bounding circle."""
        return self.vertices[0].distance(self.center)

    @property
    def area(self):
        return self.sign * (self.vertices[0].distance(self.vertices[1]) *
                            self.vertices[1].distance(self.vertices[2]))


class Square(Rectangle):
    """Square in 2D cartesian space.

    It is defined by a center and a side length.
    """

    def __init__(self, center, side_length):
        if not isinstance(center, Point):
            raise TypeError("center must be of type Point.")
        if side_length <= 0:
            raise ValueError("side_length must be greater than zero.")

        s = side_length/2
        p1 = Point([center.x + s, center.y + s])
        p2 = Point([center.x - s, center.y + s])
        p3 = Point([center.x - s, center.y - s])
        p4 = Point([center.x + s, center.y - s])
        super(Square, self).__init__(p1, p2, p3, p4)

    def __repr__(self):
        warnings.warn("The Square constructor is underdefined. The " +
                      "Rectangle constructor will be used instead.")

        return super(Square, self).__repr__()


class Mesh(Entity):
    """A collection of Entities

    Attributes
    ----------
    faces : :py:obj:`list`
        A list of the Entities
    area : float
        The total area of the Entities
    population : int
        The number entities in the Mesh
    radius : float
        The radius of a bounding circle

    """

    def __init__(self, obj=None, faces=[]):
        self.faces = []
        self.area = 0
        self.population = 0
        self.radius = 0

        if obj is not None:
            assert not faces
            self.import_triangle(obj)
        else:
            assert obj is None
            for face in faces:
                self.append(face)

    def __str__(self):
        return "Mesh(" + str(self.center) + ")"

    def __repr__(self):
        return "Mesh(faces={})".format(repr(self.faces))

    def import_triangle(self, obj):
        """Loads mesh data from a Python Triangle dict.
        """
        for face in obj['triangles']:
            p0 = Point(obj['vertices'][face[0], 0],
                       obj['vertices'][face[0], 1])
            p1 = Point(obj['vertices'][face[1], 0],
                       obj['vertices'][face[1], 1])
            p2 = Point(obj['vertices'][face[2], 0],
                       obj['vertices'][face[2], 1])
            t = Triangle(p0, p1, p2)
            self.append(t)

    @property
    def center(self):
        center = Point([0, 0])
        if self.area > 0:
            for f in self.faces:
                center += f.center * f.area
            center /= self.area
        return center

    def append(self, t):
        """Add a triangle to the mesh."""
        self.population += 1
        # self.center = ((self.center * self.area + t.center * t.area) /
        #                (self.area + t.area))
        self.area += t.area

        if isinstance(t, Polygon):
            for v in t.vertices:
                self.radius = max(self.radius, self.center.distance(v))
        else:
            self.radius = max(self.radius,
                              self.center.distance(t.center) + t.radius)

        self.faces.append(t)

    def pop(self, i=-1):
        """Pop i-th triangle from the mesh."""
        self.population -= 1
        self.area -= self.faces[i].area
        try:
            del self.__dict__['center']
        except KeyError:
            pass
        return self.faces.pop(i)

    def translate(self, vector):
        """Translate entity."""
        for t in self.faces:
            t.translate(vector)

    def rotate(self, theta, point=None, axis=None):
        """Rotate entity around an axis which passes through a point by theta
        radians."""
        for t in self.faces:
            t.rotate(theta, point, axis)

    def scale(self, vector):
        """Scale entity."""
        for t in self.faces:
            t.scale(vector)

    def contains(self, other):
        """Return whether this Mesh contains other.

        FOR ALL `x`,
        THERE EXISTS a face of the Mesh that contains `x`
        AND (ALL cut outs that contain `x` or THERE DOES NOT EXIST a cut out).
        """
        if isinstance(other, Point):
            x = other._x
        elif isinstance(other, np.ndarray):
            x = other
        elif isinstance(other, Polygon):
            x = _points_to_array(other.vertices)
            return np.all(self.contains(x))
        else:
            raise TypeError("P must be point or ndarray")

        x = np.atleast_2d(x)

        # keep track of whether each point is contained in a face
        x_in_face = np.zeros(x.shape[0], dtype=bool)
        x_in_cut = np.zeros(x.shape[0], dtype=bool)
        has_cuts = False

        for f in self.faces:
            if f.sign < 0:
                has_cuts = True
                x_in_cut = np.logical_or(x_in_cut, f.contains(x))
            else:
                x_in_face = np.logical_or(x_in_face, f.contains(x))

        if has_cuts:
            return np.logical_and(x_in_face, x_in_cut)
        else:
            return x_in_face

    @property
    def patch(self):
        patches = []
        for f in self.faces:
            patches.append(f.patch)
        return patches


def calc_standard(A):
    """Returns the standard equation (c0*x = c1) coefficents for the hyper-plane
    defined by the row-wise ND points in A. Uses single value decomposition
    (SVD) to solve the coefficents for the homogenous equations.

    Parameters
    ----------
    points : 2Darray
        Each row is an ND point.

    Returns
    ----------
    c0 : 1Darray
        The first N coeffients for the hyper-plane
    c1 : 1Darray
        The last coefficient for the hyper-plane
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("A must be np.ndarray")

    if A.ndim == 1:  # Special case for 1D
        return np.array([1]), A
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be 2D square.")

    # Add coordinate for last coefficient
    A = np.pad(A, ((0, 0), (0, 1)), 'constant', constant_values=1)

    atol = 1e-16
    rtol = 0
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T

    c = ns.squeeze()

    return c[0:-1], -c[-1]


def halfspacecirc(d, r):
    """Area of intersection between circle and half-plane. Returns the smaller
    fraction of a circle split by a line d units away from the center of the
    circle.

    Reference
    ---------
    Glassner, A. S. (Ed.). (2013). Graphics gems. Elsevier.

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
    """
    assert r > 0, "The radius must positive"
    assert d >= 0, "The distance must be positive or zero."

    if d >= r:  # The line is too far away to overlap!
        return 0

    f = 0.5 - d*sqrt(r**2 - d**2)/(np.pi*r**2) - asin(d/r)/np.pi

    # Returns the smaller fraction of the circle, so it can be at most 1/2.
    if f < 0 or 0.5 < f:
        warnings.warn("halfspacecirc was out of bounds, {}".format(f),
                      RuntimeWarning)
        f = 0

    return f
