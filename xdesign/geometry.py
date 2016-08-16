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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Point',
           'Superellipse',
           'Ellipse',
           'Circle',
           'Line',
           'Segment',
           'Ray',
           'Triangle',
           'Rectangle',
           'Square',
           'Polygon']


class Entity(object):

    """Base class for all geometric entities."""

    def __init__(self):
        pass

    @property
    def equation(self):
        """Analytical equation of the entity."""
        raise NotImplementedError

    @property
    def list(self):
        """Return list representation."""
        raise NotImplementedError

    @property
    def numpy(self):
        """Return Numpy representation."""
        return np.array(list)

    def translate(self, dx, dy):
        """Translate entity."""
        raise NotImplementedError

    def rotate(self, theta, point):
        """Rotate entity around a point."""
        raise NotImplementedError

    def scale(self, val):
        """Scale entity."""
        raise NotImplementedError

    def collision(self, entity):
        """Check if entity collides with another entity."""
        raise NotImplementedError

    def distance(self, entity):
        """Return the closest distance between entities."""
        raise NotImplementedError


class Point(Entity):

    """Point in 2-D cartesian space.

    Attributes
    ----------
    x : scalar
    y : scalar
    """

    def __init__(self, x, y):
        super(Point, self).__init__()
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, point):
        return (self.x, self.y) == (point.x, point.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, point):
        """Addition."""
        return Point(self.x + point.x, self.y + point.y)

    def __sub__(self, point):
        """Subtraction."""
        return Point(self.x - point.x, self.y - point.y)

    def __mul__(self, c):
        """Scalar multiplication."""
        return Point(c * self.x, c * self.y)

    @property
    def equation(self):
        return "(%s, %s)" % (self.x, self.y)

    @property
    def list(self):
        """Return list representation."""
        return [self.x, self.y]

    @property
    def norm(self):
        """Return the norm of the point."""
        return np.hypot(self.x, self.y)

    def translate(self, dx, dy):
        """Translate."""
        self.x += dx
        self.y += dy

    def rotate(self, theta, point=None):
        """Rotate around a point."""
        if point is None:
            point = Point(0, 0)
        dx = self.x - point.x
        dy = self.y - point.y
        px = dx * np.cos(theta) - dy * np.sin(theta)
        py = dx * np.sin(theta) + dy * np.cos(theta)
        self.x = px + point.x
        self.y = py + point.y

    def distance(self, point):
        """Return the distance from a point."""
        return np.hypot(self.x - point.x, self.y - point.y)

    def midpoint(self, point):
        """Return the midpoint from a point."""
        return self.distance(point) / 2.


class LinearEntity(Entity):

    """Base class for linear entities in 2-D Cartesian space.

    Attributes
    ----------
    p1 : Point
    p2 : Point
    """

    def __init__(self, p1, p2):
        if p1 == p2:
            raise ValueError('Requires two unique points.')
        self.p1 = p1
        self.p2 = p2

    @property
    def vertical(self):
        """True if line is vertical."""
        if self.p1.x == self.p2.x:
            return True
        else:
            return False

    @property
    def horizontal(self):
        """True if line is horizontal."""
        if self.p1.y == self.p2.y:
            return True
        else:
            return False

    @property
    def slope(self):
        """Return the slope of the line."""
        if self.vertical:
            return np.inf
        else:
            return (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)

    @property
    def equation(self):
        """Return line equation."""
        raise NotImplementedError

    @property
    def list(self):
        """Return list representation."""
        return [self.p1.x, self.p1.y, self.p2.x, self.p2.y]

    @property
    def points(self):
        """The two points used to define this linear entity."""
        return (self.p1, self.p2)

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the
        bounding rectangle for the geometric figure.
        """
        verts = self.points
        xs = [p.x for p in verts]
        ys = [p.y for p in verts]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def tangent(self):
        """Return unit tangent vector."""
        length = self.p1.distance(self.p2)
        dx = (self.p1.x - self.p2.x) / length
        dy = (self.p1.y - self.p2.y) / length
        return Point(dx, dy)

    @property
    def normal(self):
        """Return unit normal vector."""
        length = self.p1.distance(self.p2)
        dx = (self.p1.x - self.p2.x) / length
        dy = (self.p1.y - self.p2.y) / length
        return Point(-dy, dx)

    def translate(self, dx, dy):
        """Translate."""
        self.p1 = translate(self.p1, dx, dy)
        self.p2 = translate(self.p2, dx, dy)

    def rotate(self, theta, point=Point(0, 0)):
        """Rotate around a point."""
        self.p1 = rotate(self.p1, theta, point)
        self.p2 = rotate(self.p2, theta, point)


class Line(LinearEntity):

    """Line in 2-D cartesian space.

    It is defined by two distinct points.

    Attributes
    ----------
    p1 : Point
    p2 : Point
    """

    def __init__(self, p1, p2):
        super(Line, self).__init__()
        if p1 == p2:
            raise ValueError('Requires two unique points.')
        self.p1 = p1
        self.p2 = p2

    def __eq__(self, line):
        return (self.slope, self.yintercept) == (line.slope, line.yintercept)

    @property
    def xintercept(self):
        """Return the x-intercept."""
        if self.horizontal:
            return 0.
        else:
            return self.p1.x - 1 / self.slope * self.p1.y

    @property
    def yintercept(self):
        """Return the y-intercept."""
        if self.vertical:
            return 0.
        else:
            return self.p1.y - self.slope * self.p1.x

    @property
    def equation(self):
        """Return line equation."""
        if self.vertical:
            return "x = %s" % self.p1.x
        return "y = %sx + %s" % (self.slope, self.yintercept)

    def translate(self, dx, dy):
        """Translate."""
        self.p1 = translate(self.p1, dx, dy)
        self.p2 = translate(self.p2, dx, dy)

    def rotate(self, theta, point=Point(0, 0)):
        """Rotate around a point."""
        self.p1 = rotate(self.p1, theta, point)
        self.p2 = rotate(self.p2, theta, point)


class Ray(LinearEntity):

    """Ray in 2-D cartesian space.

    It is defined by two distinct points.

    Attributes
    ----------
    p1 : Point (source)
    p2 : Point (point direction)
    """

    def __init__(self, p1, p2):
        super(Line, self).__init__()
        if p1 == p2:
            raise ValueError('Requires two unique points.')
        self.p1 = p1
        self.p2 = p2

    @property
    def source(self):
        """The point from which the ray emanates."""
        return self.p1

    @property
    def direction(self):
        """The direction in which the ray emanates."""
        return self.p2 - self.p1


class Segment(LinearEntity):

    """Segment in 2-D cartesian space.

    It is defined by two distinct points.

    Attributes
    ----------
    p1 : Point (source)
    p2 : Point (point direction)
    """

    def __init__(self, p1, p2):
        super(Line, self).__init__()
        if p1 == p2:
            raise ValueError('Requires two unique points.')
        self.p1 = p1
        self.p2 = p2

    @property
    def length(self):
        """The length of the line segment."""
        return Point.distance(self.p1, self.p2)

    @property
    def midpoint(self):
        """The midpoint of the line segment."""
        return Point.midpoint(self.p1, self.p2)


class CurvedEntity(Entity):

    """Base class for curved entities in 2-D cartesian space.

    Attributes
    ----------
    center : Point
    """

    def __init__(self, center):
        super(CurvedEntity, self).__init__()
        self.center = center

    @property
    def equation(self):
        """Return analytical equation."""
        raise NotImplementedError

    @property
    def list(self):
        """Return list representation."""
        raise NotImplementedError

    @property
    def numpy(self):
        """Return Numpy representation."""
        raise NotImplementedError

    def translate(self, dx, dy):
        """Translate."""
        self.center = translate(self.center, dx, dy)

    def rotate(self, theta, point=Point(0, 0)):
        """Rotate around a point."""
        self.center = rotate(self.center, theta, point)

    def scale(self, val):
        """Scale."""
        raise NotImplementedError


class Superellipse(CurvedEntity):

    """Superellipse in 2-D cartesian space.

    Attributes
    ----------
    center : Point
    a : scalar
    b : scalar
    n : scalar
    """

    def __init__(self, center, a, b, n):
        super(Superellipse, self).__init__()
        self.center = center
        self.a = float(a)
        self.b = float(b)
        self.n = float(n)

    @property
    def list(self):
        """Return list representation."""
        return [self.center.x, self.center.y, self.a, self.b, self.n]

    @property
    def numpy(self):
        """Return Numpy representation."""
        return np.array(self.list)

    def scale(self, val):
        """Scale."""
        self.a *= val
        self.b *= val


class Ellipse(CurvedEntity):

    """Ellipse in 2-D cartesian space.

    Attributes
    ----------
    center : Point
    a : scalar
    b : scalar
    """

    def __init__(self, center, a, b):
        super(Ellipse, self).__init__()
        self.center = center
        self.a = float(a)
        self.b = float(b)

    @property
    def list(self):
        """Return list representation."""
        return [self.center.x, self.center.y, self.a, self.b]

    @property
    def numpy(self):
        """Return Numpy representation."""
        return np.array(self.list)

    def scale(self, val):
        """Scale."""
        self.a *= val
        self.b *= val


class Circle(CurvedEntity):

    """Circle in 2-D cartesian space.

    Attributes
    ----------
    center : Point
        Defines the center point of the circle.
    radius : scalar
        Radius of the circle.
    """

    def __init__(self, center, radius):
        super(Circle, self).__init__()
        self.center = center
        self.radius = float(radius)

    def __eq__(self, circle):
        return (self.x, self.y, self.radius) == (circle.x, circle.y, circle.radius)

    @property
    def equation(self):
        """Return analytical equation."""
        return "(x-%s)^2 + (y-%s)^2 = %s^2" % (self.center.x, self.center.y, self.radius)

    @property
    def list(self):
        """Return list representation."""
        return [self.center.x, self.center.y, self.radius]

    @property
    def area(self):
        """Return area."""
        return np.pi * self.radius**2

    @property
    def circumference(self):
        """Return circumference."""
        return 2 * np.pi * self.radius

    @property
    def diameter(self):
        """Return diameter."""
        return 2 * self.radius

    def scale(self, val):
        """Scale."""
        self.rad *= val


class Polygon(Entity):

    """Polygon in 2-D cartesian space.

    It is defined by n number of distinct points.

    Attributes
    ----------
    vertices : sequence of Points
    """

    def __init__(self, vertices):
        super(Polygon, self).__init__()
        self.vertices = vertices
        self.n = len(self.vertices)

    @property
    def area(self):
        """Return the area of the entity."""
        raise NotImplementedError

    @property
    def perimeter(self):
        """Return the perimeter of the entity."""
        perimeter = 0
        points = self.vertices + [self.vertices[0]]
        for i in range(self.n):
            perimeter += points[i].distance(points[i + 1])
        return perimeter


class Triangle(Polygon):

    """Triangle in 2-D cartesian space.

    It is defined by three distinct points.
    """

    def __init__(self, vertices):
        super(Triangle, self).__init__()
        if len(vertices) != 3:
            raise ValueError("Triangle requires three points.")
        self.vertices = vertices
        self.n = 3


class Rectangle(Polygon):

    """Rectangle in 2-D cartesian space.

    It is defined by four distinct points.
    """

    def __init__(self, vertices):
        super(Rectangle, self).__init__()
        if len(vertices) != 4:
            raise ValueError("Rectangle requires four points.")
        self.vertices = vertices
        self.n = 4


class Square(Polygon):

    """Square in 2-D cartesian space.

    It is defined by four distinct points.
    """

    def __init__(self, vertices):
        super(Rectangle, self).__init__()
        if len(vertices) != 2:
            raise ValueError("Square requires two points.")
        self.vertices = vertices
        self.n = 4


def rotate(point, theta, origin):
    """Rotates a point in counter-clockwise around another point.
    Parameters
    ----------
    point : Point
        An arbitrary point.
    theta : scalar
        Rotation angle in radians.
    origin : Point
        The origin of rotation axis.
    Returns
    -------
    Point
        Rotated point.
    """
    dx = point.x - origin.x
    dy = point.y - origin.y
    px = dx * np.cos(theta) - dy * np.sin(theta)
    py = dx * np.sin(theta) + dy * np.cos(theta)
    return Point(px + origin.x, py + origin.y)


def translate(point, dx, dy):
    """Translate point."""
    point.x += dx
    point.y += dy


def segment(circle, x):
    """Calculates intersection area of a vertical line segment in a circle.

    Parameters
    ----------
    circle : Circle
    x : scalar
        Intersection of the vertical line with x-axis.

    Returns
    -------
    scalar
        Area of the left region.
    """
    return circle.radius**2 * \
        np.arccos(x / circle.radius) - x * np.sqrt(circle.radius**2 - x**2)


def beamcirc(beam, circle):
    """Intersection area of an infinite beam with a circle.

    Parameters
    ----------
    beam : Beam
    circle : Circle

    Returns
    -------
    scalar
        Area of the intersected region.
    """

    # Passive coordinate transformation.
    _center = rotate(
        point=circle.center,
        theta=-np.arctan(beam.slope),
        origin=Point(0, beam.yintercept))

    # Correction if line is vertical to x-axis.
    if beam.vertical:
        dy = beam.p1.x
    else:
        dy = -beam.yintercept

    # Calculate the area deending on how the beam intersects the circle.
    p1 = _center.y - beam.size / 2. + dy
    p2 = _center.y + beam.size / 2. + dy
    pmin = min(abs(p1), abs(p2))
    pmax = max(abs(p1), abs(p2))
    if pmin < circle.radius:
        if pmax >= circle.radius:
            if p1 * p2 > 0:
                area = segment(circle, pmin)
            else:
                area = circle.area - segment(circle, pmin)
        elif pmax < circle.radius:
            area = abs(segment(circle, p1) - segment(circle, p2))
    elif p1 * p2 < 0:
        area = circle.area
    else:
        area = 0.
    return area
