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
__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'Phantom',
    'save_phantom',
    'load_phantom',
    'pickle_phantom',
    'unpickle_phantom',
]

from copy import deepcopy
import itertools
import logging
import pickle
import warnings

import numpy as np
from scipy.spatial import Delaunay

from xdesign.constants import PI
from xdesign.geometry import *
from xdesign.material import *

logger = logging.getLogger(__name__)


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
            repr(self.geometry), repr(self.children), repr(self.material)
        )

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
                if (
                    grandchild.geometry is None
                    and not contains_children(boundary, grandchild)
                ):
                    return False
                if not boundary.contains(grandchild.geometry):
                    return False
            return True

        if (
            boundary is None
            or (child.geometry is None and contains_children(boundary, child))
            or boundary.contains(child.geometry)
        ):

            child.parent = self
            self.children.append(child)
            self.population += child.population + 1
            return True

        else:
            warnings.warn(
                "{} not appended; it is not a subset.".format(repr(child)),
                RuntimeWarning
            )
            return False

    def pop(self, i=-1):
        """Pop the i-th child from the Phantom."""
        self.children[i].parent = None
        self.population -= self.children[i].population + 1
        return self.children.pop(i)

    def sprinkle(
        self,
        counts,
        radius,
        gap=0,
        region=None,
        material=None,
        max_density=1,
        shape=Circle
    ):
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
            raise ValueError(
                'Radius range must be larger than zero and largest' +
                'radius must be listed first.'
            )
        if gap < 0:
            # Support for partially overlapping phantoms is not yet supported
            # in the aquisition module
            raise NotImplementedError
        if max_density < 0:
            raise ValueError("Cannot stop at negative density.")

        collision = False
        if radius[0] + gap < 0:  # prevents circles with negative radius
            collision = True

        kTERM_CRIT = 500  # tries to append a new circle before quitting
        n_tries = 0  # attempts to append a new circle
        n_added = 0  # circles successfully added

        if region is None:
            if self.geometry is None:
                return 0
            region = self.geometry

        while (
            n_tries < kTERM_CRIT and n_added < counts
            and self.density < max_density
        ):
            center = _random_point(region, margin=radius[0])

            if collision:
                self.append(
                    Phantom(
                        geometry=Circle(center, radius[0]), material=material
                    )
                )
                n_added += 1
                continue

            circle = shape(center, radius=radius[0] + gap)
            overlap = _collision(self, circle)

            if np.max(overlap) <= radius[0] - radius[1]:

                candidate = shape(center, radius=radius[0] - np.max(overlap))
                # print(center, radius[0], gap, overlap)
                # assert np.all(_collision(self, candidate) <= 0), repr(candidate)

                self.append(Phantom(geometry=candidate, material=material))
                n_added += 1
                n_tries = 0

            n_tries += 1

        if n_added != counts and n_tries == kTERM_CRIT:
            warnings.warn((
                "Reached termination criteria of {} attempts " +
                "before adding all of the circles."
            ).format(kTERM_CRIT), RuntimeWarning)
            # no warning for reaching max_density because that's settable
        return n_added


def _collision(phantom, entity):
    """Return the max overlap of the entity and a child of this Phantom.

        May return overlap < 0; the distance between the two non-overlapping
        circles.
        """

    max_overlap = 0

    for child in phantom.children:
        if child.geometry is None:
            # get overlap from grandchildren
            overlap = _collision(child, entity)

        else:
            # calculate overlap with entity
            if isinstance(entity, Circle):
                dx = child.center.distance(entity.center)
                dr = child.radius + entity.radius

            else:
                dx = np.abs(child.center._x - entity.center._x)
                dr = (child.geometry.side_lengths + entity.side_lengths) / 2

        overlap = dr - dx

        if np.all(overlap > 0):
            max_overlap = np.maximum(max_overlap, overlap)

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
        random_point = Point([x, y])

    elif isinstance(geometry, Circle):
        radius = geometry.radius
        center = geometry.center
        r = (radius - margin) * np.sqrt(np.random.uniform(0, 1))
        a = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(a) + center.x
        y = r * np.sin(a) + center.y
        random_point = Point([x, y])

    elif isinstance(geometry, NOrthotope):
        xmin, xmax = geometry.bounding_box
        x = np.random.uniform(xmin + margin, xmax - margin)
        random_point = Point(x)
    else:
        raise NotImplementedError(
            "Cannot give point in {}.".format(type(geometry))
        )

    return random_point
