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
import matplotlib.pyplot as plt
from matplotlib.path import Path
import polytope as pt

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
           'Mesh']


class Entity(object):
    """Base class for all geometric entities. All geometric entities should
    have these methods."""

    def __init__(self):
        pass

    def translate(self, dx, dy):
        """Translate entity."""
        raise NotImplementedError

    def rotate(self, theta, point):
        """Rotate entity around a point."""
        raise NotImplementedError

    def scale(self, val):
        """Scale entity."""
        raise NotImplementedError

    def contains(self, x, y):
        """Returns true if the points are contained by the entity"""
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

    def __truediv__(self, c):
        """Scalar division."""
        return Point(self.x / c, self.y / c)

    def __str__(self):
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
    def __str__(self):
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
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
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
        super(Line, self).__init__(p1, p2)
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
    def __str__(self):
        """Return line equation."""
        if self.vertical:
            return "x = %s" % self.p1.x
        return "y = %sx + %s" % (self.slope, self.yintercept)

    @property
    def standard(self):
        """Returns coeffients for standard equation"""
        A = self.p1.y-self.p2.y
        B = self.p2.x-self.p1.x
        C = B*self.p1.y + A*self.p1.x
        return A, B, C

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
        super(Ray, self).__init__(p1, p2)
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
        super(Segment, self).__init__(p1, p2)
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
    """Base class for entities whose surface can be defined by a continuous
    equation.

    Attributes
    ----------
    center : Point
    """

    def __init__(self, center):
        super(CurvedEntity, self).__init__()
        self.center = center

    @property
    def __str__(self):
        """Return analytical equation."""
        raise NotImplementedError

    @property
    def list(self):
        """Return list representation."""
        raise NotImplementedError

    @property
    def numpy(self):
        """Return Numpy representation."""
        return np.array(self.list)

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
        super(Superellipse, self).__init__(center)
        self.center = center
        self.a = float(a)
        self.b = float(b)
        self.n = float(n)

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
        self.center = center
        self.a = float(a)
        self.b = float(b)

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


class Circle(Ellipse):
    """Circle in 2-D cartesian space.

    Attributes
    ----------
    center : Point
        Defines the center point of the circle.
    radius : scalar
        Radius of the circle.
    """

    def __init__(self, center, radius):
        super(Circle, self).__init__(center, radius, radius)
        self.center = center
        self.radius = float(radius)

    def __eq__(self, circle):
        return (self.x, self.y, self.radius) == (circle.x, circle.y,
                                                 circle.radius)

    @property
    def __str__(self):
        """Return analytical equation."""
        return "(x-%s)^2 + (y-%s)^2 = %s^2" % (self.center.x, self.center.y,
                                               self.radius)

    @property
    def list(self):
        """Return list representation."""
        return [self.center.x, self.center.y, self.radius]

    @property
    def circumference(self):
        """Return circumference."""
        return 2 * np.pi * self.radius

    @property
    def diameter(self):
        """Return diameter."""
        return 2 * self.radius

    @property
    def patch(self):
        return plt.Circle((self.center.x, self.center.y), self.radius)

    def scale(self, val):
        """Scale."""
        self.rad *= val

    def contains(self, px, py):
        return (((px-self.center.x)**2 + (py-self.center.y)**2) <=
                self.radius**2)


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
        return np.array(self.list)

    @property
    def area(self):
        """Return the area of the entity."""
        raise NotImplementedError

    @property
    def perimeter(self):
        """Return the perimeter of the entity."""
        perimeter = 0
        verts = self.vertices
        points = verts + [verts[0]]
        for m in range(self.numverts):
            perimeter += points[m].distance(points[m + 1])
        return perimeter

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the
        bounding rectangle for the geometric figure.
        """
        xs = [p.x for p in self.vertices]
        ys = [p.y for p in self.vertices]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def patch(self):
        return plt.Polygon(self.numpy)

    def contains(self, px, py):
        points = np.vstack((px.flatten(), py.flatten())).transpose()
        border = Path(self.numpy)
        bools = border.contains_points(points)
        return np.reshape(bools, px.shape)

    @property
    def half_space(self):
        """Returns the half space polytope respresentation of the polygon."""
        A = []
        B = []

        for i in range(0, self.numverts):
            edge = Line(self.vertices[i], self.vertices[(i+1) % self.numverts])
            a, b, c = edge.standard

            # test for positive or negative side of line
            if self.center.x*a + self.center.y*b > c:
                a, b, c = -a, -b, -c

            A += [a, b]
            B += [c]

        A = np.array(np.reshape(A, (self.numverts, 2)))
        B = np.array(B)
        p = pt.Polytope(A, B)
        return p


class Triangle(Polygon):
    """Triangle in 2-D cartesian space.

    It is defined by three distinct points.
    """

    def __init__(self, p1, p2, p3):
        verts = [p1, p2, p3]
        super(Triangle, self).__init__(verts)
        self.vertices = verts

    @property
    def center(self):
        center = Point(0, 0)
        for v in self.vertices:
            center += v
        return center / 3

    @property
    def area(self):
        A = self.vertices[0] - self.vertices[1]
        B = self.vertices[0] - self.vertices[2]
        return 1/2 * np.abs(np.cross([A.x, A.y], [B.x, B.y]))


class Rectangle(Polygon):
    """Rectangle in 2-D cartesian space.

    It is defined by four distinct points.
    """

    def __init__(self, p1, p2, p3, p4):
        verts = [p1, p2, p3, p4]
        super(Rectangle, self).__init__(verts)
        self.vertices = verts


class Square(Rectangle):

    """Square in 2-D cartesian space.

    It is defined by four distinct points.
    """

    def __init__(self, p1, p2, p3, p4):
        verts = [p1, p2, p3, p4]
        super(Rectangle, self).__init__(verts)
        self.vertices = verts


class Mesh(Entity):
    """A mesh object. It is a collection of polygons"""

    def __init__(self):
        self.faces = []
        self.area = 0
        self.population = 0
        self.radius = 0

    @property
    def center(self):
        center = Point(0, 0)
        if self.area > 0:
            for f in self.faces:
                center += f.center * f.area
            center /= self.area
        return center

    def append(self, t):
        """Add a triangle to the mesh."""
        assert(isinstance(t, Polygon))
        self.population += 1
        # self.center = ((self.center * self.area + t.center * t.area) /
        #                (self.area + t.area))
        self.area += t.area
        for v in t.vertices:
            self.radius = max(self.radius, self.center.distance(v))
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

    def translate(self, dx, dy):
        """Translate entity."""
        for t in self.faces:
            t.translate(dx, dy)

    def rotate(self, theta, point):
        """Rotate entity around a point."""
        for t in self.faces:
            t.rotate(theta, point)

    def scale(self, val):
        """Scale entity."""
        for t in self.faces:
            t.scale(value)

    def collision(self, entity):
        """Check if entity collides with another entity."""
        raise NotImplementedError

    def distance(self, entity):
        """Return the closest distance between entities."""
        raise NotImplementedError

    def contains(self, px, py):
        bools = np.full(px.shape, False, dtype=bool)
        for f in self.faces:
            bools = np.logical_or(bools, f.contains(px, py))
        return bools

    @property
    def patch(self):
        patches = []
        for f in self.faces:
            patches.append(f.patch)
        return patches


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


def beamintersect(beam, geometry):
    """Intersection area of infinite beam with a geometry"""
    if isinstance(geometry, Mesh):
        return beammesh(beam, geometry)
    elif isinstance(geometry, Polygon):
        return beampoly(beam, geometry)
    else:
        return beamcirc(beam, geometry)


def beammesh(beam, mesh):
    """Intersection area of infinite beam with polygonal mesh"""
    total = 0
    for face in mesh.faces:
        total += beampoly(beam, face)

    return total


def beampoly(beam, poly):
    """Intersection area of an infinite beam with a polygon"""
    return beam.half_space.intersect(poly.half_space).volume


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
