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
           'Phantom',
           'Probe',
           'sinogram']


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

    def __mul__(self, c):
        """Scalar multiplication."""
        return Point(c * self.x, c * self.y)

    def distance(self, point):
        """Returns the distance from a point."""
        dx = self.x - point.x
        dy = self.y - point.y
        return np.sqrt(dx * dx + dy * dy)

    def rotate(self, theta):
        """Returns a point rotated around origin."""
        px = self.x * np.cos(theta) - self.y * np.sin(theta)
        py = self.x * np.sin(theta) + self.y * np.cos(theta)
        return Point(px, py)

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
        """Returns an equation (string) describing the circle."""
        return "(x-%s)^2 + (y-%s)^2 = %s^2" % (self.center.x, self.center.y, self.radius)

    @property
    def area(self):
        """Returns the area of the circle."""
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
        if p1 == p2:
            raise ValueError('Requires two unique points.')
        self.p1 = p1
        self.p2 = p2

    @property
    def slope(self):
        if self.vertical:
            return np.inf
        else:
            return (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)

    @property
    def intercept(self):
        if self.vertical:
            return 0.
        else:
            return self.p1.y - self.slope * self.p1.x

    @property
    def vertical(self):
        if self.p1.x == self.p2.x:
            return True
        else:
            return False

    @property
    def horizontal(self):
        if self.p1.y == self.p2.y:
            return True
        else:
            return False

    def __str__(self):
        return "%s, %s, %s" % (self.p1, self.p2)

    def normal(self):
        length = self.p1.distance(self.p2)
        dx = (self.p1.x - self.p2.x) / length
        dy = (self.p1.y - self.p2.y) / length
        return Point(-dy, dx)

    def equation(self):
        """Returns a y-intercept equation (string) describing the line."""
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
    size : scalar, optional
        Size of the beam.
    """

    def __init__(self, p1, p2, size=0):
        super(Beam, self).__init__(p1, p2)
        self.size = float(size)

    def __str__(self):
        return super(Beam, self).__str__()


def translate(point, dx, dy):
    """Translates a point in space.

    Parameters
    ----------
    point : Point
        A point to be translated.
    dx : scalar
        Translation in x-axis.
    dy : scalar
        Translation in y-axis.

    Returns
    -------
    Point
        Translated point.
    """
    return Point(point.x + dx, point.y + dy)


def rotate(point, theta, origin=Point(0, 0)):
    """Rotates a point in counter-clockwise around an axis.

    Parameters
    ----------
    point : Point
        A point to be rotated.
    theta : scalar
        Rotation angle in radians.
    origin : Point, optional
        The location of the rotation axis.

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
        A point to be scaled.
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
    _center = rotate(
        point=circle.center,
        theta=-np.arctan(beam.slope),
        origin=Point(0, beam.intercept))

    # Correction if line is vertical to x-axis.
    if beam.vertical:
        dy = beam.p1.x
    else:
        dy = -beam.intercept

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
                area = circle.area() - segment(circle, pmin)
        elif pmax < circle.radius:
            area = abs(segment(circle, p1) - segment(circle, p2))
    elif p1 * p2 < 0:
        area = circle.area()
    else:
        area = 0.
    return area


class Phantom():
    """Phantom generation class.

    Attributes
    ----------
    shape : string
        Outline shape of the phantom. Available options: circle, square.
    population : scalar
        Number of generated circles in the phantom.
    density : scalar
        Sum of the areas of the circles in the phantom.
    feature : list
        List of circles in the phantom.
    """

    def __init__(self):
        self.shape = 'circle'
        self.population = 0
        self.density = 0
        self.feature = []

    def list(self):
        for m in range(self.population):
            print ("%s: %s" % (m, self.feature[m]))

    def add(self, circle):
        """Add a circle to the phantom.

        Parameters
        ----------
        circle : Circle
        """
        self.feature.append(circle)
        self.density += circle.area()
        self.population += 1

    def remove(self):
        """Remove last added circle from the phantom."""
        self.population -= 1
        self.density -= self.feature[-1].area()
        self.feature.pop()

    def generate_random_point(self, margin=0):
        """Generates a random point in the phantom.

        Parameters
        ----------
        margin : scalar
            Determines the margin value of the phantom.
            Points will not be created within the margin distance from the
            edge of the phantom.

        Returns
        -------
        Point
            A random point inside the phantom.
        """
        if self.shape == 'square':
            x = np.random.uniform(margin, 1 - margin)
            y = np.random.uniform(margin, 1 - margin)
        elif self.shape == 'circle':
            r = np.random.uniform(0, 0.5 - margin)
            a = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(a) + 0.5
            y = r * np.sin(a) + 0.5
        return Point(x, y)

    def sprinkle(self, counts, radius, gap=0, collision=False):
        """Randomly adds a number of circles to the phantom.

        Parameters
        ----------
        counts : int
            Number of circles to be added.
        gap : float, optional
            Minimum gap between the circle boundaries.
        collision : bool, optional
            False if circles will be non overlapping.
        """
        for m in range(int(counts)):
            center = self.generate_random_point(radius)
            circle = Circle(center, radius + gap)
            if collision or not self.collision(circle):
                self.add(Circle(center, radius))

    def collision(self, circle):
        """Checks if a circle is collided with others."""
        for m in range(self.population):
            dx = self.feature[m].center.x - circle.center.x
            dy = self.feature[m].center.y - circle.center.y
            dr = self.feature[m].radius + circle.radius
            if np.sqrt(dx**2 + dy**2) < dr:
                return True
        return False

    def plot(self):
        """Plots the generated circles."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        f = plt.figure(figsize=(8, 8), facecolor='w')
        a = f.add_subplot(111, aspect='equal')
        for m in range(self.population):
            a.add_patch(
                patches.Circle(
                    (self.feature[m].center.x,
                     self.feature[m].center.y),
                    self.feature[m].radius,
                    transform=a.transAxes))
        plt.grid('on')
        plt.show()

    def save(self, filename):
        pass


class Probe(Beam):

    def __init__(self, p1, p2, size=0):
        super(Probe, self).__init__(p1, p2, size)

    def translate(self, dx):
        """Translates beam along its normal direction."""
        vec = self.normal() * dx
        self.p1 += vec
        self.p2 += vec

    def rotate(self, theta, origin):
        """Rotates beam around a given point."""
        self.p1 = rotate(self.p1, theta, origin)
        self.p2 = rotate(self.p2, theta, origin)

    def measure(self, phantom):
        newdata = 0
        for m in range(phantom.population):
            newdata += beamcirc(self, phantom.feature[m])
        return newdata


def sinogram(sx, sy, phantom):
    """Generates sinogram given a phantom.

    Parameters
    ----------
    sx : int
        Number of samples in the rotation direction
    sy : int
        Number of samples in the y direction
    phantom : Phantom

    """
    size = 1. / sy
    beta = np.pi / sx
    sino = np.zeros((sx, sy))
    p = Probe(Point(size / 2., 0), Point(size / 2., 1), size)
    for m in range(sx):
        print (m, m * beta * 180. / np.pi)
        for n in range(sy):
            sino[m, n] = p.measure(phantom)
            p.translate(size)
        p.translate(-1)
        p.rotate(beta, Point(0.5, 0.5))
    return sino
