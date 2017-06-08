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

"""Defines an object for simulating X-ray phantoms.

.. moduleauthor:: Daniel J Ching <carterbox@users.noreply.github.com>
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
import warnings
from copy import deepcopy
from scipy.spatial import Delaunay
import pickle
import itertools

from xdesign.geometry import *
from xdesign.material import *
from xdesign.constants import PI

logger = logging.getLogger(__name__)


__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Phantom',
           'save_phantom',
           'load_phantom',
           'pickle_phantom',
           'unpickle_phantom',
           'XDesignDefault',
           'HyperbolicConcentric',
           'DynamicRange',
           'DogaCircles',
           'SlantedSquares',
           'UnitCircle',
           'Soil',
           'WetCircles',
           'SiemensStar',
           'Foam',
           'Metal',
           'SoftBiomaterial',
           'Electronics',
           'FiberComposite',
           'Softwood']


def save_phantom(phantom, filename):
    """Save phantom to file as a python repr."""
    f = open(filename, 'w')
    f.write("{}".format(repr(phantom)))
    f.close()
    logger.info('Save Phantom to {}'.format(filename))


def load_phantom(filename):
    """Load phantom from file containing a python repr."""
    f = open(filename, 'r')
    raw_phantom = f.read()
    f.close()
    logger.info('Load Phantom from {}'.format(filename))
    return eval(raw_phantom)


def pickle_phantom(phantom, filename):
    """Save phantom to file as a python pickle."""
    f = open(filename, 'wb')
    pickle.dump(phantom, f)


def unpickle_phantom(filename):
    """Load phantom from file as a python pickle."""
    f = open(filename, 'rb')
    return pickle.load(f)


class Phantom(object):
    """An object for the purpose of evaluating X-ray imaging methods.

    Phantoms may be hierarchical structures with children that are contained
    within and/or a parent which contains them. They have two parts: a geometry
    and properties. The geometry defines the spatial extent over which the
    properties are valid. Properties are parameters which a :class:`.Probe`
    uses to measure the Phantom.

    All Phantoms must fit within the geometry of their ancestors. Phantoms
    whose geometry is None act as containers.

    Attributes
    ----------
    geometry : :class:`.Entity`
        The spatial boundary of the Phantom; may be None.
    children :
        A list of Phantoms contained in this Phantom.
    parent :
        The Phantom containing this Phantom.
    material :
        The mass_attenuation of the phantom.
    population :
        The number of decendents of this phantom.
    """
    # OPERATOR OVERLOADS
    def __init__(self, geometry=None, children=[], material=None):

        self._geometry = geometry
        self.population = 0
        self.parent = None
        self.material = material

        self.children = list()
        for child in children:
            self.append(child)

    def __add__(self, other):
        """Combine two Phantoms."""
        parent = Phantom()
        parent.append(self)
        parent.append(other)
        return parent

    def __str__(self):
        return "{}()".format(type(self).__name__)

    def __repr__(self):
        return "Phantom(geometry={}, children={}, material={})".format(
                repr(self.geometry),
                repr(self.children),
                repr(self.material))

    # PROPERTIES
    @property
    def is_leaf(self):
        """Return whether the Phantom is a leaf node."""
        return not self.children

    @property
    def geometry(self):
        """Return the geometry of the Phantom."""
        return self._geometry

    @property
    def center(self):
        """Return the centroid of the Phantom."""
        if self.geometry is None:
            return None

        return self.geometry.center

    @property
    def radius(self):
        """Return the radius of the smallest boundary sphere."""
        if self.geometry is None:
            return None

        return self.geometry.radius

    @property
    def volume(self):
        """Return the volume of the Phantom"""
        if self.geometry is None:
            return None

        if hasattr(self.geometry, 'volume'):
            return self.geometry.volume
        else:
            return self.geometry.area

    @property
    def density(self):
        '''Return the geometric density of the Phantom.'''
        if self.geometry is None:
            return None

        child_volume = 0

        for child in self.children:
            child_volume += child.volume

        return child_volume / self.volume

    # GEOMETRIC TRANSFORMATIONS
    def translate(self, vector):
        """Translate the Phantom."""
        for child in self.children:
            child.translate(vector)

        if self._geometry is not None:
            self._geometry.translate(vector)

    def rotate(self, theta, point=Point([0.5, 0.5]), axis=None):
        """Rotate around an axis that passes through the given point."""
        for child in self.children:
            child.rotate(theta, point, axis)

        if self._geometry is not None:
            self.geometry.rotate(theta, point, axis)

    # TREE MANIPULATION
    def append(self, child):
        """Add a child to the Phantom.

        Only add the child if it is contained within the geometry of its
        ancestors.
        """
        boundary = self.geometry
        parent = self.parent

        while boundary is None and parent is not None:
                boundary = parent.geometry
                parent = parent.parent

        def contains_children(boundary, child):
            for grandchild in child.children:
                if (grandchild.geometry is None
                        and not contains_children(boundary, grandchild)):
                    return False
                if not boundary.contains(grandchild.geometry):
                    return False
            return True

        if (boundary is None
                or (child.geometry is None and contains_children(boundary,
                                                                 child))
                or boundary.contains(child.geometry)):

            child.parent = self
            self.children.append(child)
            self.population += child.population + 1
            return True

        else:
            warnings.warn("{} not appended; it is not a subset.".format(
                          repr(child)), ValueError)
            return False

    def pop(self, i=-1):
        """Pop the i-th child from the Phantom."""
        self.children[i].parent = None
        self.population -= self.children[i].population + 1
        return self.children.pop(i)

    def sprinkle(self, counts, radius, gap=0, region=None,
                 material=None, max_density=1):
        """Sprinkle a number of :class:`.Circle` shaped Phantoms around the
        Phantom. Uses various termination criteria to determine when to stop
        trying to add circles.

        Parameters
        ----------
        counts : int
            The number of circles to be added.
        radius : scalar or list
            The radius of the circles to be added.
        gap : float, optional
            The minimum distance between circle boundaries.
            A negative value allows overlapping edges.
        region : :class:`.Entity`, optional
            The new circles are confined to this shape. None if the circles are
            allowed anywhere.
        max_density : scalar, optional
            Stops adding circles when the geometric density of the phantom
            reaches this ratio.
        material : scalar, optional
            A mass attenuation parameter passed to the circles.

        Returns
        ----------
        counts : scalar
            The number of circles successfully added.
        """
        if counts < 0:
            raise ValueError('Cannot add negative number of circles.')
        if not isinstance(radius, list):
            radius = [radius, radius]
        if len(radius) != 2 or radius[0] < radius[1] or radius[1] <= 0:
            raise ValueError('Radius range must be larger than zero and largest' +
                       'radius must be listed first.')
        if gap < 0:
            # Support for partially overlapping phantoms is not yet supported
            # in the aquisition module
            raise NotImplementedError
        if max_density < 0:
            raise ValueError("Cannot stop at negative density.")

        collision = False
        if radius[0] + gap < 0:  # prevents circles with negative radius
            collision = True

        kTERM_CRIT = 200  # tries to append a new circle before quitting
        n_tries = 0  # attempts to append a new circle
        n_added = 0  # circles successfully added

        if region is None:
            if self.geometry is None:
                return 0
            region = self.geometry

        while (n_tries < kTERM_CRIT and n_added < counts and
               self.density < max_density):
            center = _random_point(region, margin=radius[0])

            if collision:
                self.append(Phantom(geometry=Circle(center, radius[0]),
                                    material=material))
                n_added += 1
                continue

            circle = Circle(center, radius[0] + gap)
            overlap = _collision(self, circle)
            if overlap <= radius[0] - radius[1]:
                self.append(Phantom(geometry=Circle(center,
                                                    radius[0] - overlap),
                                    material=material))
                n_added += 1
                n_tries = 0

            n_tries += 1

        if n_added != counts and n_tries == kTERM_CRIT:
            warnings.warn(("Reached termination criteria of {} attempts " +
                           "before adding all of the circles.").format(
                           kTERM_CRIT), RuntimeWarning)
            # no warning for reaching max_density because that's settable
        return n_added


def _collision(phantom, circle):
        """Return the max overlap of the circle and a child of this Phantom.

        May return overlap < 0; the distance between the two non-overlapping
        circles.
        """
        max_overlap = 0

        for child in phantom.children:
            if child.geometry is None:
                overlap = _collision(child, circle)

            else:
                dx = child.center.distance(circle.center)
                dr = child.radius + circle.radius
                overlap = dr - dx

            max_overlap = max(max_overlap, overlap)

        return max_overlap


def _random_point(geometry, margin=0.0):
    """Return a Point located within the geometry.

    Parameters
    ----------
    margin : scalar
        Determines the margin value of the shape.
        Points will not be created in the margin area.

    """
    if isinstance(geometry, Rectangle):
        [xmin, ymin, xmax, ymax] = geometry.bounds
        x = np.random.uniform(xmin + margin, xmax - margin)
        y = np.random.uniform(ymin + margin, ymax - margin)

    elif isinstance(geometry, Circle):
        radius = geometry.radius
        center = geometry.center
        r = (radius - margin) * np.sqrt(np.random.uniform(0, 1))
        a = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(a) + center.x
        y = r * np.sin(a) + center.y

    else:
        raise NotImplementedError("Cannot give point in {}.".format(
                                  type(geometry)) + " Only Square and " +
                                  "Circle are available.")

    return Point([x, y])


class XDesignDefault(Phantom):
    """Generates a Phantom for internal testing of XDesign.

    The default phantom is:
    nested, it contains phantoms within phantoms;
    geometrically simple, the sinogram can be verified visually; and
    representative, it contains the three main geometric elements: circle,
        polygon, and mesh.
    """

    def __init__(self):
        super(XDesignDefault, self).__init__(geometry=Circle(Point([0.5, 0.5]),
                                                             radius=0.5),
                                             material=SimpleMaterial(0.0))

        # define the points of the mesh
        a = Point([0.6, 0.6])
        b = Point([0.6, 0.4])
        c = Point([0.8, 0.4])
        d = (a + c) / 2
        e = (a + b) / 2

        t0 = Triangle(deepcopy(b), deepcopy(c), deepcopy(d))

        # construct and reposition the mesh
        m0 = Mesh()
        m0.append(Triangle(deepcopy(a), deepcopy(e), deepcopy(d)))
        m0.append(Triangle(deepcopy(b), deepcopy(d), deepcopy(e)))

        # define the circles
        m1 = Mesh()
        m1.append(Circle(Point([0.3, 0.5]), radius=0.1))
        m1.append(-Circle(Point([0.3, 0.5]), radius=0.02))

        # construct Phantoms
        self.append(Phantom(children=[Phantom(geometry=t0,
                                              material=SimpleMaterial(0.5)),
                                      Phantom(geometry=m0,
                                              material=SimpleMaterial(0.5))]))
        self.append(Phantom(geometry=m1, material=SimpleMaterial(1.0)))


class HyperbolicConcentric(Phantom):
    """Generates a series of cocentric alternating black and white circles whose
    radii are changing at a parabolic rate. These line spacings cover a range
    of scales and can be used to estimate the Modulation Transfer Function. The
    radii change according to this function: r(n) = r0*(n+1)^k.

    Attributes
    ----------
    radii : list
        The list of radii of the circles
    widths : list
        The list of the widths of the bands
    """

    def __init__(self, min_width=0.1, exponent=1/2):
        """
        Parameters
        ----------
        min_width : scalar
            The radius of the smallest ring in the series.
        exponent : scalar
            The exponent in the function r(n) = r0*(n+1)^k.
        """

        super(HyperbolicConcentric, self).__init__()
        center = Point([0.5, 0.5])
        Nmax_rings = 512

        radii = [0]
        widths = [min_width]
        for ring in range(0, Nmax_rings):
            radius = min_width * np.power(ring + 1, exponent)
            if radius > 0.5 and ring % 2:
                break

            self.append(Phantom(geometry=Circle(center, radius),
                                material=SimpleMaterial((-1.)**(ring % 2))))
            # record information about the rings
            widths.append(radius - radii[-1])
            radii.append(radius)

        self.children.reverse()  # smaller circles on top
        self.radii = radii
        self.widths = widths


class DynamicRange(Phantom):
    """Generates a phantom of randomly placed circles for determining dynamic
    range.

    Parameters
    -------------
    steps : scalar, optional
        The orders of magnitude (base 2) that the colors of the circles cover.
    jitter : bool, optional
        True : circles are placed in a jittered grid
        False : circles are randomly placed
    shape : string, optional
    """

    def __init__(self, steps=10, jitter=True,
                 geometry=Square(center=Point([0.5, 0.5]), side_length=1)):
        super(DynamicRange, self).__init__(geometry=geometry)

        # determine the size and and spacing of the circles around the box.
        spacing = 1.0 / np.ceil(np.sqrt(steps))
        radius = spacing / 4

        colors = [2.0**j for j in range(0, steps)]
        np.random.shuffle(colors)

        if jitter:
            # generate grid
            _x = np.arange(0, 1, spacing) + spacing / 2
            px, py, = np.meshgrid(_x, _x)
            px = np.ravel(px)
            py = np.ravel(py)

            # calculate jitters
            jitters = 2 * radius * (np.random.rand(2, steps) - 0.5)

            # place the circles
            for i in range(0, steps):
                center = Point([px[i] + jitters[0, i], py[i] + jitters[1, i]])
                self.append(Phantom(geometry=Circle(center, radius),
                                    material=SimpleMaterial(colors[i])))
        else:
            # completely random
            for i in range(0, steps):
                if 1 > self.sprinkle(1, radius, gap=radius * 0.9,
                                     material=SimpleMaterial(colors[i])):
                    None
                    # TODO: ensure that all circles are placed


class DogaCircles(Phantom):
    """Rows of increasingly smaller circles. Initally arranged in an ordered
    Latin square, the inital arrangement can be randomly shuffled.

    Attributes
    ----------
    radii : ndarray
        radii of circles
    x : ndarray
        x position of circles
    y : ndarray
        y position of circles
    """
    # IDEA: Use method in this reference to calculate uniformly distributed
    # latin squares.
    # DOI: 10.1002/(SICI)1520-6610(1996)4:6<405::AID-JCD3>3.0.CO;2-J
    def __init__(self, n_sizes=5, size_ratio=0.5, n_shuffles=5):
        """
        Parameters
        ----------
        n_sizes : int
            number of different sized circles
        size_ratio : scalar
            the nth size / the n-1th size
        n_shuffles : int
            The number of times to shuffles the latin square
        """
        super(DogaCircles, self).__init__(geometry=Square(center=Point([0.5,
                                                                        0.5]),
                                                          side_length=1))

        n_sizes = int(n_sizes)
        if n_sizes <= 0:
            raise ValueError('There must be at least one size.')
        if size_ratio > 1 or size_ratio <= 0:
            raise ValueError('size_ratio should be <= 1 and > 0.')
        n_shuffles = int(n_shuffles)
        if n_shuffles < 0:
            raise ValueError('Cant shuffle a negative number of times')

        # Seed a latin square, use integers to prevent rounding errors
        top_row = np.array(range(0, n_sizes), dtype=int)
        rowsum = np.sum(top_row)
        lsquare = np.empty([n_sizes, n_sizes], dtype=int)
        for i in range(0, n_sizes):
            lsquare[:, i] = np.roll(top_row, i)

        # Choose a row or column shuffle sequence
        sequence = np.random.randint(0, 2, n_shuffles)

        # Shuffle the square
        for dim in sequence:
            lsquare = np.rollaxis(lsquare, dim, 0)
            np.random.shuffle(lsquare)

        # Assert that it is still a latin square.
        for i in range(0, n_sizes):
            assert np.sum(lsquare[:, i]) == rowsum, \
                "Column {0} is {1} and should be {2}".format(i, np.sum(
                                                        lsquare[:, i]), rowsum)
            assert np.sum(lsquare[i, :]) == rowsum, \
                "Column {0} is {1} and should be {2}".format(i, np.sum(
                                                        lsquare[i, :]), rowsum)

        # Draw it
        period = np.arange(0, n_sizes)/n_sizes + 1/(2*n_sizes)
        _x, _y = np.meshgrid(period, period)
        radii = 1/(2*n_sizes)*size_ratio**lsquare

        for (k, x, y) in zip(radii.flatten(), _x.flatten(),
                             _y.flatten()):
            self.append(Phantom(geometry=Circle(Point([x, y]), k),
                                material=SimpleMaterial(1.0)))

        self.radii = radii
        self.x = _x
        self.y = _y


class SlantedSquares(Phantom):
    """Generates a collection of slanted squares. Squares are arranged in
    concentric circles such that the space between squares is at least gap. The
    size of the squares is adaptive such that they all remain within the unit
    circle.

    Attributes
    ----------
    angle : scalar
        the angle of slant in radians
    count : scalar
        the total number of squares
    gap : scalar
        the minimum space between squares
    side_length : scalar
        the size of the squares
    squares_per_level : list
        the number of squares at each level
    radius_per_level : list
        the radius at each level
    n_levels : scalar
        the number of levels
    """

    def __init__(self, count=10, angle=5/360*2*PI, gap=0):
        super(SlantedSquares, self).__init__()
        if count < 1:
            raise ValueError("There must be at least one square.")

        # approximate the max diameter from total area available
        d_max = np.sqrt(PI/4 / (2 * count))

        if 1 < count and count < 5:
            # bump all the squares to the 1st ring and calculate sizes
            # as if there were 5 total squares
            pass

        while True:
            squares_per_level = [1]
            radius_per_level = [0]
            remaining = count - 1
            n_levels = 1
            while remaining > 0:
                # calculate next level capacity
                radius_per_level.append(radius_per_level[n_levels-1] + d_max +
                                        gap)
                this_circumference = PI*2*radius_per_level[n_levels]
                this_capacity = this_circumference//(d_max + gap)

                # assign squares to levels
                if remaining - this_capacity >= 0:
                    squares_per_level.append(this_capacity)
                    remaining -= this_capacity
                else:
                    squares_per_level.append(remaining)
                    remaining = 0
                n_levels += 1
                assert(remaining >= 0)

            # Make sure squares will not be outside the phantom, else
            # decrease diameter by 5%
            if radius_per_level[-1] < (0.5 - d_max/2 - gap):
                break
            d_max *= 0.95

        assert(len(squares_per_level) == len(radius_per_level))

        # determine center positions of squares
        x, y = np.array([]), np.array([])
        for level in range(0, n_levels):
            radius = radius_per_level[level]
            thetas = (((np.arange(0, squares_per_level[level]) /
                      squares_per_level[level]) +
                      1/(squares_per_level[level] * 2)) *
                      2 * PI)
            x = np.concatenate((x, radius*np.cos(thetas)))
            y = np.concatenate((y, radius*np.sin(thetas)))

        # move to center of phantom.
        x += 0.5
        y += 0.5

        # add the squares to the phantom
        side_length = d_max/np.sqrt(2)
        for i in range(0, x.size):
            center = Point([x[i], y[i]])
            s = Square(center=center, side_length=side_length)
            s.rotate(angle, center)
            self.append(Phantom(geometry=s, material=SimpleMaterial(1)))

        self.angle = angle
        self.count = count
        self.gap = gap
        self.side_length = side_length
        self.squares_per_level = squares_per_level
        self.radius_per_level = radius_per_level
        self.n_levels = n_levels


class UnitCircle(Phantom):
    """Generates a phantom with a single circle in its center."""

    def __init__(self, radius=0.5, material=SimpleMaterial(1.0)):
        super(UnitCircle, self).__init__(geometry=Circle(Point([0.5, 0.5]),
                                                         radius),
                                         material=material)


class Soil(UnitCircle):
    """Generates a phantom with structure similar to soil.

    References
    -----------
    Schlüter, S., Sheppard, A., Brown, K., & Wildenschild, D. (2014). Image
    processing of multiphase images obtained via X‐ray microtomography: a
    review. Water Resources Research, 50(4), 3615-3639.
    """

    def __init__(self, porosity=0.412):
        super(Soil, self).__init__(radius=0.5, material=SimpleMaterial(0.5))
        self.sprinkle(30, [0.1, 0.03], 0, material=SimpleMaterial(0.5),
                      max_density=1-porosity)
        # use overlap to approximate area opening transform because opening is
        # not discrete
        self.sprinkle(100, 0.02, 0.01, material=SimpleMaterial(-.25))


class WetCircles(UnitCircle):
    def __init__(self):
        super(WetCircles, self).__init__(radius=0.5,
                                         material=SimpleMaterial(0.5))
        porosity = 0.412
        np.random.seed(0)

        self.sprinkle(30, [0.1, 0.03], 0.005, material=SimpleMaterial(0.5),
                      max_density=1 - porosity)

        pairs = [(23, 12), (12, 19), (29, 11), (22, 5), (1, 3), (21, 9),
                 (8, 2), (2, 27)]
        for p in pairs:
            A = self.children[p[0]-1].geometry
            B = self.children[p[1]-1].geometry

            thetaA = [PI/2, 10]
            thetaB = [PI/2, 10]

            mesh = wet_circles(A, B, thetaA, thetaB)

            self.append(Phantom(geometry=mesh, material=-.25))


def wet_circles(A, B, thetaA, thetaB):
    """Generates a mesh that wets the surface of circles A and B.

    Parameters
    -------------
    A,B : Circle
    theta : list
        the number of radians that the wet covers and number of the points on
        the surface range
    """

    vector = B.center - A.center
    if vector.x > 0:
        angleA = np.arctan(vector.y/vector.x)
        angleB = PI + angleA
    else:
        angleB = np.arctan(vector.y/vector.x)
        angleA = PI + angleB
    # print(vector)
    rA = A.radius
    rB = B.radius

    points = []
    for t in ((np.arange(0, thetaA[1])/(thetaA[1]-1) - 0.5)
              * thetaA[0] + angleA):

        x = rA*np.cos(t) + A.center.x
        y = rA*np.sin(t) + A.center.y
        points.append([x, y])

    mid = len(points)
    for t in ((np.arange(0, thetaB[1])/(thetaB[1]-1) - 0.5)
              * thetaB[0] + angleB):

        x = rB*np.cos(t) + B.center.x
        y = rB*np.sin(t) + B.center.y
        points.append([x, y])

    points = np.array(points)

    # Triangulate the polygon
    tri = Delaunay(points)

    # Remove extra triangles
    # print(tri.simplices)
    mask = np.sum(tri.simplices < mid, 1)
    mask = np.logical_and(mask < 3, mask > 0)
    tri.simplices = tri.simplices[mask, :]
    # print(tri.simplices)

    m = Mesh()
    for t in tri.simplices:
        m.append(Triangle(Point([points[t[0], 0], points[t[0], 1]]),
                          Point([points[t[1], 0], points[t[1], 1]]),
                          Point([points[t[2], 0], points[t[2], 1]])))

    return m


class SiemensStar(Phantom):
    """Generates a Siemens star.

    Attributes
    ----------
    ratio : scalar
        The spatial frequency times the proportional radius. e.g to get the
        frequency, f, divide this ratio by some fraction of the maximum radius:
        f = ratio/radius_fraction
    """
    def __init__(self, n_sectors=4, center=Point([0.5, 0.5]), radius=0.5):
        """
        Parameters
        ----------
        n_sectors: int >= 4
            The number of spokes/blades on the star.
        center: Point
        radius: scalar > 0
        """
        super(SiemensStar, self).__init__()
        if n_sectors < 4:
            raise ValueError("Must have >= 4 sectors.")
        if radius <= 0:
            raise ValueError("radius must be greater than zero.")
        if not isinstance(center, Point):
            raise TypeError("center must be of type Point.!")
        n_points = n_sectors

        # generate an even number of points around the unit circle
        points = []
        for t in (np.arange(0, n_points)/n_points) * 2 * PI:
            x = radius*np.cos(t) + center.x
            y = radius*np.sin(t) + center.y
            points.append(Point([x, y]))
        assert(len(points) == n_points)

        # connect pairs of points to the center to make triangles
        for i in range(0, n_sectors//2):
            f = Phantom(geometry=Triangle(points[2*i], points[2*i+1], center),
                        material=SimpleMaterial(1))
            self.append(f)

        self.ratio = n_points / (4 * PI * radius)
        self.n_sectors = n_sectors


class Foam(UnitCircle):
    """Generates a phantom with structure similar to foam."""

    def __init__(self, size_range=[0.05, 0.01], gap=0, porosity=1):
        super(Foam, self).__init__(radius=0.5, material=SimpleMaterial(1.0))
        if porosity < 0 or porosity > 1:
            raise ValueError('Porosity must be in the range [0,1).')
        self.sprinkle(300, size_range, gap, material=SimpleMaterial(-1.0),
                      max_density=porosity)


class Softwood(Phantom):
    """Generate a Phantom with structure similar to wood.

    Parameters
    ----------
    ringsize : float [cm]
        The thickness of the annual rings in cm.
    latewood_fraction : float
        The volume ratio of latewood cells to earlywood cells
    ray_fraction : float
        The ratio of rows of ray cells to rows of tracheids
    ray_height : float [cm]
        The height of the ray cells
    cell_width, cell_height : float [cm]
        The shape of the earlywood cells
    cell_thickness : float [cm]
        The thickness of the earlywood cell walls
    frame : arraylike [cm]
        A bounding box for the cells
    """

    def __init__(self):
        super(Softwood, self).__init__()

        ring_size = 0.5
        latewood_fraction = 0.35

        ray_fraction = 1/8
        ray_height = 0.01
        ray_width = 0.09
        ray_thickness = 0.002

        cell_width, cell_height = 0.03, 0.03
        cell_thickness = 0.004

        frame = np.array([[0.2, 0.2], [0.8, 0.8]])

        # -------------------
        def five_p():
            return 1 + np.random.normal(scale=0.05)

        atol = 1e-16  # for rounding errors
        cellulose = SimpleMaterial(1)

        x0, y0 = frame[0, 0], frame[0, 1]
        x1, y1 = frame[1, 0], frame[1, 1]

        # Place the cells one by one at (x, y)
        y = y0
        for r in itertools.count():
            # Check that the row is in the frame
            if y + cell_height > y1 and abs(y + cell_height - y1) > atol:
                # Stop if cell reaches out of frame
                break

            # Add random jitter to each row
            x = x0 + cell_width * np.random.normal(scale=0.1)
            if r % 2 == 1:
                # Offset odd number rows by 1/2 cell width
                x += cell_width / 2

            # Decide whether to make a ray cell
            if np.random.rand() < ray_fraction:
                is_ray = True
            else:
                is_ray = False

            ring_progress = 0
            for c in itertools.count():

                if x < x0 and abs(x - x0) > atol:
                    # skip first cell if jittered outside the frame
                    x += cell_width

                if is_ray:
                    cell = WoodCell(corner=Point([x, y]), material=cellulose,
                                    width=ray_width * five_p(),
                                    height=ray_height,
                                    wall_thickness=ray_thickness * five_p())

                else:  # not ray cells
                    if ring_progress < 1 - latewood_fraction:
                        # TODO: Add some randomness to when latewood starts
                        dw, dt = 1, 1
                    else:
                        # transition to latewood
                        dw = 0.6
                        dt = 1.5

                    cell = WoodCell(corner=Point([x, y]), material=cellulose,
                                    width=cell_width * dw * five_p(),
                                    height=cell_height,
                                    wall_thickness=cell_thickness * dt * five_p())
                self.append(cell)

                x += cell.width
                ring_progress = (x / ring_size) % 1

                if x + cell.width > x1 and abs(x + cell.width - x1) > atol:
                    # Stop if cell reaches out of frame
                    break

            y += cell.height


class WoodCell(Phantom):
    """Generate a Phantom with structure similar to a single wood cell.

    A wood cell has two parts: the lumen which is the empty center area of the
    cell and the cell wall substance which is generally hexagonal.
    """

    def __init__(self, corner=Point([0.5, 0.5]), width=0.003, height=0.003,
                 wall_thickness=0.0008, material=None):
        super(WoodCell, self).__init__()

        p1 = deepcopy(corner)
        p2 = deepcopy(corner) + Point([width, 0])
        p3 = deepcopy(corner) + Point([width, height])
        p4 = deepcopy(corner) + Point([0, height])
        cell_wall = Rectangle(p1, p2, p3, p4)

        wt = wall_thickness
        p1 = deepcopy(corner) + Point([wt, wt])
        p2 = deepcopy(corner) + Point([width - wt, wt])
        p3 = deepcopy(corner) + Point([width - wt, height - wt])
        p4 = deepcopy(corner) + Point([wt, height-wt])
        lumen = -Rectangle(p1, p2, p3, p4)

        self._geometry = Mesh(faces=[cell_wall, lumen])
        self.material = material
        self.height = height
        self.width = width
        # self.append(center)


class Metal(Phantom):

    def __init__(self, shape='square'):
        raise NotImplementedError


class SoftBiomaterial(Phantom):

    def __init__(self, shape='square'):
        raise NotImplementedError


class Electronics(Phantom):

    def __init__(self, shape='square'):
        raise NotImplementedError


class FiberComposite(Phantom):

    def __init__(self, shape='square'):
        raise NotImplementedError
