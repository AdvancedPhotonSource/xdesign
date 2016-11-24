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

from xdesign.geometry import *
from xdesign.geometry import Entity
from xdesign.feature import *
import numpy as np
import scipy.ndimage
import logging
import warnings

logger = logging.getLogger(__name__)


__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Phantom']


class Phantom(object):
    """Phantoms are objects for the purpose of evaluating an imaging method.

    Each Phantom is a square or circular region containing a :class:`.list` of :class:`.Feature` objects. The :mod:`.acquisition` module uses Phantoms as an interface for generating data.

    Phantoms can be combined using the '+' operator, and they also have some of the same mehtods as the :class:`.List` class including: append, pop, insert, sort, and reverse.

    Attributes
    ----------
    shape : :class:`str`
        The shape of the phantom: circle, square.
    population : scalar
        The number of :class:`.Feature` in the Phantom.
    area : scalar
        The total volume of the :class:`.Feature` in the Phantom.
    feature : :class:`list`
        List of :class:`.Feature`.
    """
    # OPERATOR OVERLOADS
    def __init__(self, shape='circle'):
        if not (shape == 'circle' or shape == 'square'):
            raise ValueError("Phantom must be a circle or square.")
        self.shape = shape
        self.population = 0
        self.area = 0
        self.feature = []

    def __add__(self, other):
        if not isinstance(other, Phantom):
            raise TypeError("Can only add phantoms to other phantoms.")
        self.population += other.population
        self.area += other.area
        self.feature += other.feature
        return self

    # PROPERTIES
    @property
    def list(self):
        """Prints the contents of the Phantom."""
        for m in range(self.population):
            print(self.feature[m].list)

    @property
    def density(self):
        '''Returns the area density of the phantom. Does not acount for
        functional weight of the Features.
        '''
        if self.shape == 'square':
            return self.area
        elif self.shape == 'circle':
            return self.area / (np.pi * 0.5 * 0.5)

    # FEATURE LIST MANIPULATION
    def append(self, feature):
        """Add a Feature to the top of the phantom."""
        if not isinstance(feature, Feature):
            raise TypeError("Can only add Features to Phantoms.")
        self.feature.append(feature)
        self.area += feature.area
        self.population += 1

    def pop(self, i=-1):
        """Pop the i-th Feature from the Phantom."""
        self.population -= 1
        self.area -= self.feature[i].area
        return self.feature.pop(i)

    def insert(self, i, feature):
        """Insert a Feature at a given depth."""
        if not isinstance(feature, Feature):
            raise TypeError("Can only add Features to Phantoms.")
        self.feature.insert(i, feature)
        self.area += feature.area
        self.population += 1

    def sort(self, param="mass_atten", reverse=False):
        """Sorts the Features by a property such as mass_atten or size."""
        if param == "mass_atten":
            def key(feature): return feature.mass_atten
        elif param == "size":
            def key(feature): return feature.area
        else:
            raise ValueError("Can't sort by " + param)
        self.feature = sorted(self.feature, key=key, reverse=reverse)

    def reverse(self):
        """Reverse the order of the Features in the phantom."""
        self.feature.reverse()

    def sprinkle(self, counts, radius, gap=0, region=None, mass_atten=1,
                 max_density=1):
        """Sprinkles a number of :class:`.Circle` shaped Features around the Phantom. Uses various termination criteria to determine when to stop trying to add circles.

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
            The new circles are confined to this shape. None if the circles are allowed anywhere.
        max_density : scalar, optional
            Stops adding circles when the geometric density of the phantom reaches this ratio.
        mass_atten : scalar, optional
            A mass attenuation parameter passed to the circles.

        Returns
        ----------
        counts : scalar
            The number of circles successfully added.
        """
        if counts < 0:
            ValueError('Cannot add negative number of circles.')
        if not isinstance(radius, list):
            radius = [radius, radius]
        if len(radius) != 2 or radius[0] < radius[1] or radius[1] <= 0:
            ValueError('Radius range must be larger than zero and largest' +
                       'radius must be listed first.')
        if gap < 0:
            # Support for partially overlapping features is not yet supported
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

        while (n_tries < kTERM_CRIT and n_added < counts and
               self.density < max_density):
            center = self._random_point(radius[0], region=region)

            if collision:
                self.append(Feature(Circle(center, radius[0]),
                            mass_atten=mass_atten))
                n_added += 1
                continue

            circle = Feature(Circle(center, radius[0] + gap))
            overlap = self._collision(circle)
            if overlap <= radius[0] - radius[1]:
                self.append(Feature(Circle(center, radius[0] - overlap),
                                    mass_atten=mass_atten))
                n_added += 1
                n_tries = 0

            n_tries += 1

        if n_added != counts and n_tries == kTERM_CRIT:
            warnings.warn("Reached termination criteria of " +
                          str(kTERM_CRIT) + " attempts before adding " +
                          "all of the circles.", RuntimeWarning)
            # no warning for reaching max_density because that's settable
        return n_added

    # GEOMETRIC TRANSFORMATIONS
    def translate(self, dx, dy):
        """Translate phantom."""
        for m in range(self.population):
            self.feature[m].translate(dx, dy)

    def rotate(self, theta, origin=Point([0.5, 0.5]), axis=None):
        """
        Rotates the Phantom around an axis passing through the given origin.
        """
        for m in range(self.population):
            self.feature[m].rotate(theta, origin, axis)

    # IMPORT AND EXPORT
    def numpy(self):
        """Returns the Numpy representation."""
        # Phantoms contain more than circles now.
        arr = np.empty((self.population, 4))
        for m in range(self.population):
            arr[m] = [
                self.feature[m].center.x,
                self.feature[m].center.y,
                self.feature[m].radius,
                self.feature[m].mass_atten]
        return arr

    def save(self, filename):
        """Saves phantom to file."""
        np.savetxt(filename, self.numpy(), delimiter=',')

    def load(self, filename):
        """Load phantom from file."""
        arr = np.loadtxt(filename, delimiter=',')
        for m in range(arr.shape[0]):
            self.append(Feature(
                        Circle(Point([arr[m, 0], arr[m, 1]]), arr[m, 2]),
                        arr[m, 3]))

    # PRIVATE METHODS
    def _random_point(self, margin=0, region=None):
        """Generate a random point in the given geometric entity.

        Parameters
        ----------
        margin : scalar
            Determines the margin value of the shape.
            Points will not be created in the margin area.
        region : Entity, optional
            Determines where the point will be generated. None assumes it can
            be generated anywhere in the 1x1 phantom.

        Returns
        -------
        Point
            Random point.
        """
        if isinstance(region, Entity):
            raise NotImplementedError
        else:
            radius = 0.5
            center = Point([0.5, 0.5])

        if self.shape == 'square':
            x = np.random.uniform(margin - radius, radius - margin) + center.x
            y = np.random.uniform(margin - radius, radius - margin) + center.y
        elif self.shape == 'circle':
            r = np.random.uniform(0, radius - margin)
            a = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(a) + center.x
            y = r * np.sin(a) + center.y

        return Point([x, y])

    def _collision(self, circle):
        """Check if a circle is collided with another circle.

        Returns
        --------
        overlap : scalar
            The largest amount that the circle is overlapping
        """
        if not isinstance(circle, Feature):
            raise TypeErrorz

        overlap = 0
        for m in range(self.population):
            dx = self.feature[m].center.x - circle.center.x
            dy = self.feature[m].center.y - circle.center.y
            dr = self.feature[m].radius + circle.radius
            overlap = max(dr - np.sqrt(dx**2 + dy**2), overlap)

        return overlap
