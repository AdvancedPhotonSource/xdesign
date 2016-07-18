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
           'rotate',
           'translate',
           'scale',
           'segment',
           'beamcirc',
           'Registry']


class Point(object):
    """Point in 2-D Cartesian space.

    Attributes
    ----------
    x : scalar
    y : scalar
    """

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return "(%s, %s)" % (self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        """Addition."""
        return Point(self.x + other.x, self.y + other.y)

    def __rmul__(self, c):
        """Scalar multiplication."""
        return Point(c * self.x, c * self.y)

    def list(self):
        """Returns the point's list representation."""
        return [self.x, self.y]

    def numpy(self):
        """Returns the Numpy representation."""
        return np.array([self.x, self.y])


class Circle(object):
    """Circle in 2-D Cartesian space.

    Attributes
    ----------
    center : Point
        Center point of the circle.
    radius : scalar, optional
        Radius of the circle.
    """

    def __init__(self, center, radius):
        self.center = center
        self.radius = float(radius)

    def __str__(self):
        return "%s, %s" % (self.center, self.radius)

    def equation(self):
        return "(x-%s)^2 + (y-%s)^2 = %s^2" % (self.center.x, self.center.y, self.radius)

    def area(self):
        return np.pi * self.radius**2


class Line(object):
    """Line in 2-D Cartesian space.

    It is defined by two distinct points.

    Attributes
    ----------
    p1 : Point
    p2 : Point
    """

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

        if self.p1 == self.p2:
            raise ValueError('Requires two unique points.')

        elif self.p1.x == self.p2.x:
            self.slope = np.inf
            self.intercept = 0.
            self.vertical = True

        else:
            self.slope = (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)
            self.intercept = self.p1.y - self.slope * self.p1.x
            self.vertical = False

    def __str__(self):
        return "%s, %s, %s" % (self.p1, self.p2)

    def equation(self):
        if self.vertical:
            return "x = %s" % self.p1.x
        return "y = %sx + %s" % (self.slope, self.intercept)


class Beam(Line):
    """Beam (thick line) in 2-D Cartesian space.

    It is defined by two distinct points.

    Attributes
    ----------
    p1 : Point
    p2 : Point
    width : scalar, optional
        Width of the beam.
    """

    def __init__(self, p1, p2, width=0):
        super(Beam, self).__init__(p1, p2)
        self.width = float(width)

    def __str__(self):
        return super(Beam, self).__str__()


def translate(point, dx, dy):
    """Translates a point in space.

    Parameters
    ----------
    point : Point
        An arbitrary point.
    dx : scalar
        Translation in x-axis.
    dy : Point, optional
        Translation in y-axis.

    Returns
    -------
    Point
        Translated point.
    """
    return Point(point.x + dx, point.y + dy)


def rotate(point, theta, origin=Point(0, 0)):
    """Rotates a point in counter-clockwise around another point.

    Parameters
    ----------
    point : Point
        An arbitrary point.
    theta : scalar
        Rotation angle in radians.
    origin : Point, optional
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


def scale(point, ds):
    """Scales a point in space.

    Parameters
    ----------
    point : Point
        An arbitrary point.
    ds : scalar
        Scaling value.

    Returns
    -------
    Point
        Scaled point.
    """
    return Point(point.x * ds, point.y * ds)


def segment(circle, x):
    """Calculates intersection area of a vertical line segment in a circle.

    Parameters
    ----------
    circle : Circle
    x : scalar
        Intersection of the vertical line with x-axis.
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
    _circle = rotate(
        point=circle.center,
        theta=-np.arctan(beam.slope),
        origin=Point(0, beam.intercept))

    # Correction if line is parallel to x-axis.
    if beam.p1.y == beam.p2.y:
        dy = beam.p1.y
    else:
        dy = 0

    # Calculate the area deending on how the beam intersects the circle.
    p1 = _circle.y - beam.width / 2. + dy
    p2 = _circle.y + beam.width / 2. + dy
    pmin = min(abs(p1), abs(p2))
    pmax = max(abs(p1), abs(p2))
    if pmin < circle.radius:
        if pmax >= circle.radius:
            if p1 * p2 > 0:
                area = segment(circle, pmin)
            else:
                area = circle.area() - segment(circle, pmin)
        elif pmax < circle.radius:  #
            area = abs(segment(circle, p1) - segment(circle, p2))
    elif p1 * p2 < 0:
        area = circle.area()
    else:
        area = 0.
    return area


class Registry():
    """Bookkeeping of circles generated.

    Attributes
    ----------
    population : scalar
        Number of generated circles in the phantom.
    density : scalar
        Density of the circles in the phantom.
    feature : list
        List of circles.
    """

    def __init__(self):
        self.population = 0
        self.density = 0
        self.feature = []

    def __str__(self):
        return "%s" % self.feature

    def add(self, circle):
        """Add a circle to the phantom.

        Parameters
        ----------
        circle : Circle
        """
        circ = Circle(center, radius)
        self.feature.append(circle)
        self.density += circle.area()
        self.population += 1

    def remove(self):
        """Remove last added circle from the phantom.
        """
        self.population -= 1
        self.density -= self.feature[-1].area()
        self.feature.pop()
