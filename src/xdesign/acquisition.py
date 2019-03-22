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
"""Define objects and methods for simulated data acquisition.

This not only includes physical things like :py:class:`.Probe`, detectors,
turntables, and lenses, but non-physical things such as scanning patterns.

.. moduleauthor:: Doga Gursoy <dgursoy@aps.anl.gov>
.. moduleauthor:: Daniel J Ching <carterbox@users.noreply.github.com>
"""

__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'Probe',
    'raster_scan2D',
]

import logging

import numpy as np
from cached_property import cached_property

from xdesign.constants import RADIUS, DEFAULT_ENERGY
from xdesign.geometry import *
from xdesign.geometry.intersect import halfspacecirc, clip_SH

logger = logging.getLogger(__name__)


class Probe(Line):
    """A square cross-section x-ray beam for probing Phantoms.

    Attributes
    ----------
    p1, p2 : :py:class:`xdesign.geometry.Point`
        .. deprecated:: 0.4
            Measure now uses theta, h, v coordinates instead.
    size : float, cm (default: 0.0 cm)
        The size of probe in centimeters.
    intensity : float, cd (default: 1.0 cd)
        The intensity of the beam in candela.
    energy : float, eV (default: 15 eV)
        The energy of the probe in eV.

    .. todo::
        Implement additional attributes for Probe such as wavelength,
        etc.

    """

    def __init__(
        self, p1=None, p2=None, size=0.0, intensity=1.0, energy=DEFAULT_ENERGY
    ):
        if p1 is None or p2 is None:
            p1 = Point([RADIUS, 0])
            p2 = Point([-RADIUS, 0])
        super(Probe, self).__init__(p1, p2)
        self.size = size
        self.intensity = intensity
        self.energy = energy

    def __repr__(self):
        return "Probe({}, {}, size={}, intensity={}, energy={})".format(
            repr(self.p1), repr(self.p2), repr(self.size), repr(self.intensity),
            repr(self.energy)
        )

    def __str__(self):
        """Return the string respresentation of the Beam."""
        return "Probe(" + super(Probe, self).__str__() + ")"

    def distance(self, other):
        """Return the closest distance between entities."""
        dx = super(Probe, self).distance(other)
        return dx - self.size / 2

    def measure(self, phantom, theta, h, perc=None):
        """Measure the phantom from the given position.

        Parameters
        ----------
        theta, h
            The coordinates of the Probe.

        """
        if perc is not None:
            raise UserWarning(
                "Noise during acquisition has been removed. "
                "Use post-processing to add noise "
                "instead."
            )
        assert theta.shape == h.shape, "theta, h, must be the same shape"
        original_shape = theta.shape
        newdata = np.zeros(theta.size)
        # Convert probe coordinates to cartesian coordinates
        srcx, srcy, detx, dety = thv_to_zxy(theta.ravel(), h.ravel())
        # Calculate measurement for each position
        for i in range(theta.size):
            self.p1 = Point([srcx[i], srcy[i]])
            self.p2 = Point([detx[i], dety[i]])
            newdata[i] = (
                self.intensity * np.exp(-self._get_attenuation(phantom))
            )
        logger.debug("Probe.measure: {}".format(newdata))
        return newdata.reshape(original_shape)

    def _get_attenuation(self, phantom):
        """Return the beam intensity attenuation due to the phantom."""
        intersection = beamintersect(self, phantom.geometry)

        if intersection is None or phantom.material is None:
            attenuation = 0.0
        else:
            # [ ] = [cm^2] / [cm] * [1/cm]
            attenuation = (
                intersection / self.cross_section *
                phantom.material.linear_attenuation(self.energy)
            )

        if phantom.geometry is None or intersection > 0:
            # check the children for containers and intersecting geometries
            for child in phantom.children:
                attenuation += self._get_attenuation(child)

        return attenuation

    @property
    def cross_section(self):
        """Return the cross-sectional area of a square beam."""
        return self.size
        # return np.pi * self.size**2 / 4

    def half_space(self):
        """Return the half space polytope respresentation of the probe."""
        half_space = list()

        for i in range(2):
            edge = Line(
                self.p1 + self.normal * self.size / 2 * (-1)**i,
                self.p2 + self.normal * self.size / 2 * (-1)**i
            )
            A, B = edge.standard

            # test for positive or negative side of line
            if np.dot(self.p1._x, A) > B:
                A = -A
                B = -B

            half_space.append([A, B])

        return half_space

    def intersect(self, polygon):
        """Return the intersection with polygon."""
        return clip_SH(self.half_space(), polygon)


def thv_to_zxy(theta, h):
    """Convert coordinates from (theta, h, v) to (z, x, y) space."""
    cos_p = np.cos(theta)
    sin_p = np.sin(theta)
    srcx = +RADIUS * cos_p - h * sin_p
    srcy = +RADIUS * sin_p + h * cos_p
    detx = -RADIUS * cos_p - h * sin_p
    dety = -RADIUS * sin_p + h * cos_p
    return srcx, srcy, detx, dety


def beamintersect(beam, geometry):
    """Intersection area of infinite beam with a geometry."""

    logger.debug('BEAMINTERSECT: {}'.format(repr(geometry)))

    if geometry is None:
        return None
    elif isinstance(geometry, Mesh):
        return beammesh(beam, geometry)
    elif isinstance(geometry, Polygon):
        return beampoly(beam, geometry)
    elif isinstance(geometry, Circle):
        return beamcirc(beam, geometry)
    else:
        raise NotImplementedError


def beammesh(beam, mesh):
    """Intersection area of infinite beam with polygonal mesh."""
    if beam.distance(mesh.center) > mesh.radius:
        logger.debug("BEAMMESH: skipped because of radius.")
        return 0

    volume = 0

    for f in mesh.faces:
        volume += f.sign * beamintersect(beam, f)

    return volume


def beampoly(beam, poly):
    """Intersection area of an infinite beam with a polygon."""
    if beam.distance(poly.center) > poly.radius:
        logger.debug("BEAMPOLY: skipped because of radius.")
        return 0

    intersection = beam.intersect(poly)
    if len(intersection) > 0:
        return Polygon(intersection).area
    else:
        return 0


def beamcirc(beam, circle):
    """Intersection area of a Beam (line with finite thickness) and a circle.

    Reference
    ---------
    Glassner, A. S. (Ed.). (2013). Graphics gems. Elsevier.

    Parameters
    ----------
    beam : Beam
    circle : Circle

    Returns
    -------
    a : scalar
        Area of the intersected region.

    """
    r = circle.radius
    w = beam.size / 2
    p = super(Probe, beam).distance(circle.center)
    assert (p >= 0)

    logger.debug("BEAMCIRC: r = %f, w = %f, p = %f" % (r, w, p))

    if w == 0 or r == 0:
        return 0

    if w < r:
        if p < w:
            f = 1 - halfspacecirc(w - p, r) - halfspacecirc(w + p, r)
        elif p < r - w:  # and w <= p
            f = halfspacecirc(p - w, r) - halfspacecirc(w + p, r)
        else:  # r - w <= p
            f = halfspacecirc(p - w, r)
    else:  # w >= r
        if p < w:
            f = 1 - halfspacecirc(w - p, r)
        else:  # w <= pd
            f = halfspacecirc(p - w, r)

    a = np.pi * r**2 * f
    assert (a >= 0), a
    return a


def raster_scan2D(sa, st, meta=False):
    """Return the coordinates of a 2D raster scan.

    Parameters
    ----------
    sa : int
        The number of projeciton angles in `[0, 2PI)`.
    st : int
        The number of Probe steps at each projection angle. `[-0.5, 0.5)`
    nmeta : int >= 0
        The number of meta steps. Meta steps are the offset from the starting
        Probe position after each rotation.

    Returns
    -------
    theta, h, v : :py:class:`np.array` (M,)
        Probe positions for scan

    """
    theta = np.linspace(0, np.pi * 2, sa, endpoint=False)
    h = np.linspace(0, 1, st, endpoint=False) - 0.5
    theta, h = np.meshgrid(theta, h)
    return theta, h
