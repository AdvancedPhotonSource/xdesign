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
from scipy.spatial import Delaunay

logger = logging.getLogger(__name__)


__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Material',
           'HyperbolicConcentric',
           'DynamicRange',
           'UnitCircle',
           'WetCircles',
           'Soil',
           'Foam',
           'Electronics']


class Material(object):
    """Placeholder for class which uses NIST data to automatically calculate
    material properties based on beam energy.
    """

    def __init__(self, formula, density):
        # calculate the value based on the photon energy
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
    of scales and can be used to estimate the Modulation Transfer Function.
    """

    def __init__(self, min_width=0.1, exponent=1 / 2):
        """
        Attributes
        -------------
        radii : list
            The list of radii of the circles
        widths : list
            The list of the widths of the bands
        """
        super(HyperbolicConcentric, self).__init__(shape='circle')
        center = Point(0.5, 0.5)
        # exponent = 1/2
        Nmax_rings = 512

        radii = [0]
        widths = [min_width]
        for ring in range(0, Nmax_rings):
            radius = min_width * np.power(ring + 1, exponent)
            if radius > 0.5 and ring % 2:
                break

            self.append(Feature(
                        Circle(center, radius), value=(-1.)**(ring % 2)))
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
                            Circle(center, radius), value=colors[i]))
        else:
            # completely random
            for i in range(0, steps):
                if 1 > self.sprinkle(1, radius, gap=radius * 0.9,
                                     value=colors[i]):
                    None
                    # TODO: ensure that all circles are placed


class UnitCircle(Phantom):
    """Generates a phantom with a single circle in its center."""

    def __init__(self, radius=0.5, value=1):
        super(UnitCircle, self).__init__()
        self.append(Feature(
                    Circle(Point(0.5, 0.5), radius), value))


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
        self.sprinkle(30, [0.1, 0.03], 0, value=0.5, max_density=1 - porosity)
        # use overlap to approximate area opening transform because opening is
        # not discrete
        self.sprinkle(100, 0.02, 0.01, value=-.25)
        background = Feature(Circle(Point(0.5, 0.5), 0.5), value=0.5)
        self.insert(0, background)


class WetCircles(Phantom):
    def __init__(self):
        super(WetCircles, self).__init__(shape='circle')

        # np.random.seed(0)
        self.sprinkle(2, 0.1, 0.1)

        A = self.feature[0].geometry
        B = self.feature[1].geometry
        # A = Feature(Circle(Point(0.2,0.4),0.1))
        # B = Feature(Circle(Point(0.6,0.4),0.1))

        thetaA = [np.pi/2, 10]
        thetaB = [np.pi/2, 10]

        mesh = wet_circles(A, B, thetaA, thetaB)

        self.append(Feature(mesh))
        # self.append(A)
        # self.append(B)


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
        angleB = np.pi + angleA
    else:
        angleB = np.arctan(vector.y/vector.x)
        angleA = np.pi + angleB
    # print(vector)
    rA = A.radius
    rB = B.radius

    points = []
    for t in (np.arange(0, thetaA[1])/(thetaA[1]-1) - 0.5) * thetaA[0] + angleA:
        x = rA*np.cos(t) + A.center.x
        y = rA*np.sin(t) + A.center.y
        points.append([x, y])

    mid = len(points)
    for t in (np.arange(0, thetaB[1])/(thetaB[1]-1) - 0.5) * thetaB[0] + angleB:
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
        m.append(Triangle(Point(points[t[0], 0], points[t[0], 1]),
                          Point(points[t[1], 0], points[t[1], 1]),
                          Point(points[t[2], 0], points[t[2], 1])))

    return m


class Foam(Phantom):
    """Generates a phantom with structure similar to foam."""

    def __init__(self):
        super(Foam, self).__init__(shape='circle')
        self.sprinkle(300, [0.05, 0.01], 0, value=-1)
        background = Feature(Circle(Point(0.5, 0.5), 0.5), value=1)
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
