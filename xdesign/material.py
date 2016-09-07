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
from xdesign.phantom import *
from xdesign.geometry import *
from xdesign.feature import *
from xdesign.constants import PI

logger = logging.getLogger(__name__)


__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Material',
           'HyperbolicConcentric',
           'DynamicRange',
           'UnitCircle',
           'Soil',
           'Foam',
           'Electronics']


class Material(object):
    """Placeholder for class which uses NIST data to automatically calculate
    material properties based on beam energy.
    """

    def __init__(self, formula, density):
        # calculate the mass_atten based on the photon energy
        super(Material, self).__init__()
        self.formula = formula
        self.density = density

    @property
    def compton_cross_section(self, energy):
        """Compton cross-section of the electron [cm^2]."""
        raise NotImplementedError

    @property
    def photoelectric_cross_section(self, energy):
        raise NotImplementedError

    @property
    def atomic_form_factor(self, energy):
        """Measure of the scattering amplitude of a wave by an isolated atom.
        Read from NIST database [Unitless]."""
        raise NotImplementedError

    @property
    def atom_concentration(self, energy):
        """Number of atoms per unit volume [1/cm^3]."""
        raise NotImplementedError

    @property
    def reduced_energy_ratio(self, energy):
        """Energy ratio of the incident x-ray and the electron energy
        [Unitless]."""
        raise NotImplementedError

    @property
    def photoelectric_absorption(self, energy):
        """X-ray attenuation due to the photoelectric effect [1/cm]."""
        raise NotImplementedError

    @property
    def compton_scattering(self, energy):
        """X-ray attenuation due to the Compton scattering [1/cm]."""
        raise NotImplementedError

    @property
    def electron_density(self, energy):
        """Electron density [e/cm^3]."""
        raise NotImplementedError

    @property
    def linear_attenuation(self, energy):
        """Total x-ray attenuation [1/cm]."""
        raise NotImplementedError

    @property
    def refractive_index(self, energy):
        raise NotImplementedError

    def mass_ratio(self):
        raise NotImplementedError

    def number_of_elements(self):
        raise NotImplementedError


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

        super(HyperbolicConcentric, self).__init__(shape='circle')
        center = Point(0.5, 0.5)
        Nmax_rings = 512

        radii = [0]
        widths = [min_width]
        for ring in range(0, Nmax_rings):
            radius = min_width * np.power(ring + 1, exponent)
            if radius > 0.5 and ring % 2:
                break

            self.append(Feature(
                        Circle(center, radius), mass_atten=(-1.)**(ring % 2)))
            # record information about the rings
            widths.append(radius - radii[-1])
            radii.append(radius)

        self.reverse()  # smaller circles on top
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

    def __init__(self, steps=10, jitter=True, shape='square'):
        super(DynamicRange, self).__init__(shape=shape)

        # determine the size and and spacing of the circles around the box.
        spacing = 1. / np.ceil(np.sqrt(steps))
        radius = spacing / 4

        colors = [2**j for j in range(0, steps)]
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
                center = Point(px[i] + jitters[0, i], py[i] + jitters[1, i])
                self.append(Feature(
                            Circle(center, radius), mass_atten=colors[i]))
        else:
            # completely random
            for i in range(0, steps):
                if 1 > self.sprinkle(1, radius, gap=radius * 0.9,
                                     mass_atten=colors[i]):
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
        super(DogaCircles, self).__init__(shape='square')

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
            self.append(Feature(Circle(Point(x, y), k)))

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
        super(SlantedSquares, self).__init__(shape='circle')
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
            center = Point(x[i], y[i])
            s = Square(center, side_length)
            s.rotate(angle, center)
            self.append(Feature(s))

        self.angle = angle
        self.count = count
        self.gap = gap
        self.side_length = side_length
        self.squares_per_level = squares_per_level
        self.radius_per_level = radius_per_level
        self.n_levels = n_levels


class UnitCircle(Phantom):
    """Generates a phantom with a single circle in its center."""

    def __init__(self, radius=0.5, mass_atten=1):
        super(UnitCircle, self).__init__()
        self.append(Feature(
                    Circle(Point(0.5, 0.5), radius), mass_atten))


class Soil(Phantom):
    """Generates a phantom with structure similar to soil.

    References
    -----------
    Schlüter, S., Sheppard, A., Brown, K., & Wildenschild, D. (2014). Image
    processing of multiphase images obtained via X‐ray microtomography: a
    review. Water Resources Research, 50(4), 3615-3639.
    """

    def __init__(self, porosity=0.412):
        super(Soil, self).__init__(shape='circle')
        self.sprinkle(30, [0.1, 0.03], 0, mass_atten=0.5,
                      max_density=1-porosity)
        # use overlap to approximate area opening transform because opening is
        # not discrete
        self.sprinkle(100, 0.02, 0.01, mass_atten=-.25)
        background = Feature(Circle(Point(0.5, 0.5), 0.5), mass_atten=0.5)
        self.insert(0, background)


class Foam(Phantom):
    """Generates a phantom with structure similar to foam."""

    def __init__(self, size_range=[0.05, 0.01], gap=0, porosity=0):
        super(Foam, self).__init__(shape='circle')
        if porosity < 0 or porosity > 1:
            raise ValueError('Porosity must be in the range [0,1).')
        self.sprinkle(300, size_range, gap, mass_atten=-1,
                      max_density=1-porosity)
        background = Feature(Circle(Point(0.5, 0.5), 0.5), mass_atten=1)
        self.insert(0, background)


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
