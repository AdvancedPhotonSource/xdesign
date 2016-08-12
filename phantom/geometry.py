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
           'Circle',
           'Line',
           'Beam',
           'Feature']


class Feature(object):

    """Base feature class."""

    def __init__(self):
        pass

    @property
    def equation(self):
        """Analytical equation of the feature."""
        raise NotImplementedError

    @property
    def list(self):
        """Return list representation."""
        raise NotImplementedError

    @property
    def numpy(self):
        """Return Numpy representation."""
        raise NotImplementedError

    @property
    def area(self):
        """Return the area of the feature."""
        raise NotImplementedError

    @property
    def circumference(self):
        """Return the circumference of the feature."""
        raise NotImplementedError

    def translate(self, dx, dy):
        """Translate feature."""
        raise NotImplementedError

    def rotate(self, theta, point):
        """Rotate feature around a point."""
        raise NotImplementedError

    def scale(self, val):
        """Scale feature."""
        raise NotImplementedError

    def collision(self, feature):
        """Check if feature collides with another feature."""
        raise NotImplementedError

    def distance(self, feature):
        """Return the closest distance between features."""
        raise NotImplementedError


class Point(Feature):

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
    def numpy(self):
        """Return Numpy representation."""
        return np.array([self.x, self.y])

    @property
    def area(self):
        return 0

    @property
    def circumference(self):
        return 0

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


class Circle(Feature):

    """Circle in 2-D cartesian space.

    Attributes
    ----------
    center : Point
        Defines the center point of the circle.
    radius : scalar
        Radius of the circle.
    """

    def __init__(self, center, radius, value=1):
        super(Circle, self).__init__()
        self.center = center
        self.radius = float(radius)
        self.value = float(value)

    def __eq__(self, circle):
        return (self.x, self.y, self.radius, self.value) == (circle.x, circle.y, circle.radius, circle.value)

    @property
    def equation(self):
        """Return analytical equation."""
        return "(x-%s)^2 + (y-%s)^2 = %s^2" % (self.center.x, self.center.y, self.radius)

    @property
    def list(self):
        """Return list representation."""
        return [self.center.x, self.center.y, self.radius, self.value]

    @property
    def numpy(self):
        """Return Numpy representation."""
        return np.array(self.list)

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

    def translate(self, dx, dy):
        """Translate."""
        self.center = translate(self.center, dx, dy)

    def rotate(self, theta, point=Point(0, 0)):
        """Rotate around a point."""
        self.center = rotate(self.center, theta, point)

    def scale(self, val):
        """Scale."""
        self.rad *= val


class Line(Feature):

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

    @property
    def list(self):
        """Return list representation."""
        return [self.p1.x, self.p1.y, self.p2.x, self.p2.y]

    @property
    def numpy(self):
        """Return Numpy representation."""
        return np.array(self.list)

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


class Beam(Line):

    """Beam (thick line) in 2-D cartesian space.

    It is defined by two distinct points.

    Attributes
    ----------
    p1 : Point
    p2 : Point
    size : scalar, optional
        Size of the beam.
    """

    def __init__(self, p1, p2, size=0):
        super(Beam, self).__init__(p1, p2)
        self.size = float(size)

    def __str__(self):
        return super(Beam, self).__str__()


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
