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

"""
Defines objects and methods for simulated data acquisition.

The acquistion module contains the objects and procedures necessary to simulate
the operation of equipment used to collect tomographic data. This not only
includes physical things like Probes, detectors, turntables, and lenses, but
also non-physical things such as scanning patterns and programs.

.. moduleauthor:: Doga Gursoy <dgursoy@aps.anl.gov>
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from xdesign.geometry import *
from xdesign.geometry import halfspacecirc
import logging
import polytope as pt
from copy import deepcopy
from cached_property import cached_property

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Beam',
           'Probe',
           'sinogram',
           'angleogram',
           'raster_scan',
           'angle_scan']


class Beam(Line):
    """A thick line in 2-D cartesian space.

    A Beam is defined by two distinct points and a size (thickness). It is
    a subclass of a Probe.

    Attributes
    ----------
    p1 : Point
    p2 : Point
    size : scalar, optional, cm
        Size of the beam. i.e. the diameter
    intensity : float, optional, cd
        The intensity of the beam in candela.
    """
    # TODO: Determine whether separate Beam object is necessary or if Beam can
    # be merged with Probe.
    def __init__(self, p1, p2, size=0.0, intensity=1.0):
        """Return a new Beam from two given points and optional size."""
        super(Beam, self).__init__(p1, p2)
        self.size = float(size)
        self.intensity = intensity
        self.count = 0

    def __repr__(self):
        return "Beam({}, {}, size={})".format(repr(self.p1), repr(self.p2),
                                              repr(self.size))

    def __str__(self):
        """Return the string respresentation of the Beam."""
        return "Beam(" + super(Beam, self).__str__() + ")"

    def distance(self, other):
        """Return the closest distance between entities."""
        dx = super(Beam, self).distance(other)
        return dx - self.size / 2

    def translate(self, vector):
        """Translate entity along vector."""
        logger.info("Translating Beam.")
        self.p1.translate(vector)
        self.p2.translate(vector)

        if 'half_space' in self.__dict__:
            self.half_space = self.half_space.translation(vector)

    def rotate(self, theta, point=None, axis=None):
        """Rotate entity around an axis which passes through an point by theta
        radians."""
        logger.info("Rotating Beam.")
        self.p1.rotate(theta, point, axis)
        self.p2.rotate(theta, point, axis)

        if 'half_space' in self.__dict__:
            if point is None:
                d = 0
            else:
                d = point._x

            self.half_space = self.half_space.translation(-d)
            self.half_space = self.half_space.rotation(0, 1, theta)
            self.half_space = self.half_space.translation(d)

    @cached_property
    def half_space(self):
        """Return the half space polytope respresentation of the infinite
        beam."""
        # add half beam width along the normal direction to each of the points
        half = self.normal * self.size / 2
        edges = [Line(self.p1 + half, self.p2 + half),
                 Line(self.p1 - half, self.p2 - half)]

        A = np.ndarray((len(edges), self.dim))
        B = np.ndarray(len(edges))

        for i in range(0, 2):
            A[i, :], B[i] = edges[i].standard

            # test for positive or negative side of line
            if np.einsum('i, i', self.p1._x, A[i, :]) > B[i]:
                A[i, :] = -A[i, :]
                B[i] = -B[i]

        p = pt.Polytope(A, B)
        return p

    @property
    def skip(self):
        self.count += 1
        return self.count


def beamintersect(beam, geometry):
    """Intersection area of infinite beam with a geometry"""

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
    """Intersection area of infinite beam with polygonal mesh"""
    if beam.distance(mesh.center) > mesh.radius:
        logger.info("BEAMMESH skipped because of radius.")
        return 0

    volume = 0

    for f in mesh.faces:
        volume += f.sign * beamintersect(beam, f)

    return volume


def beampoly(beam, poly):
    """Intersection area of an infinite beam with a polygon"""
    if beam.distance(poly.center) > poly.radius:
        logger.info("BEAMPOLY skipped because of radius.")
        return 0

    return beam.half_space.intersect(poly.half_space).volume


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
    w = beam.size/2
    p = super(Beam, beam).distance(circle.center)
    assert(p >= 0)

    logger.info("BEAMCIRC r = %f, w = %f, p = %f" % (r, w, p))

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
    assert(a >= 0), a
    return a


class Probe(Beam):
    """An object for probing Phantoms.

    A Probe provides an interface for measuring the interaction of a Phantom
    and a beam. It contains information for interacting with Materials such as
    energy and brightness.

    Attributes
    -----------------
    size : float, cm
        The size of probe in centimeters.
    intensity : float, cd
        The intensity of the beam in candela.
    """
    # TODO: Implement additional attributes for Probe such as beam energy,
    # brightness, wavelength, etc.
    def __init__(self, p1, p2, size=0.0, intensity=1.0, energy=15.0):
        super(Probe, self).__init__(p1, p2, size, intensity)
        self.energy = energy
        self.history = []

    def __repr__(self):
        return "Probe({}, {}, size={})".format(repr(self.p1), repr(self.p2),
                                               repr(self.size))

    def translate(self, dx):
        """Translate beam along its normal direction."""
        vec = self.normal * dx
        super(Probe, self).translate(vec._x)

    def measure(self, phantom, sigma=0.0, pool=None):
        """Return the probe measurement with optional Gaussian noise.

        Parameters
        ----------
        sigma : float >= 0
            The standard deviation of the normally distributed noise.
        """
        newdata = self.intensity * np.exp(-self._get_attenuation(phantom))

        if sigma > 0:
            newdata += newdata * np.random.normal(scale=sigma)

        self.record()
        return newdata

    def _get_attenuation(self, phantom):
        """Return the beam intensity attenuation due to the phantom."""
        intersection = beamintersect(self, phantom.geometry)

        if intersection is None or phantom.material is None:
            attenuation = 0.0
        else:
            # [ ] = [cm^2] / [cm] * [1/cm]
            attenuation = (intersection / self.size
                           * phantom.material.linear_attenuation(self.energy))

        if phantom.geometry is None or intersection > 0:
            # check the children for containers and intersecting geometries
            for child in phantom.children:
                attenuation += self._get_attenuation(child)

        return attenuation

    def record(self):
        self.history.append(self.list)


def probe_wrapper(probes, phantom, noise):
    """Wrap probe.measure to make it suitable for multiprocessing.

    This method does two things: (1) it puts the Probe.measure method in the
    f(*args) format for the pool workers (2) it passes chunks of work (multiple
    probes) to the workers to reduce overhead.
    """

    for i in range(len(probes)):
        probes[i] = probes[i].measure(phantom, noise)

    return probes


def calculate_gram(procedure, measurements, phantom, noise=0.0, pool=None,
                   chunksize=1):
    """This part of the code is identical for angleogram and sinogram."""

    if pool is None:
        sx = measurements.size

        for m in range(sx):
            probe = next(procedure)
            measurements[m] = probe.measure(phantom, noise)

    else:
        sx = measurements.size//chunksize
        sy = chunksize
        measurements.shape = (sx, sy)

        async_data = [None] * sx
        for m in range(sx):

            probes = [None] * sy
            for n in range(sy):

                probes[n] = deepcopy(next(procedure))

            async_data[m] = pool.apply_async(probe_wrapper,
                                             (probes, phantom, noise,))

        for m in range(sx):
            measurements[m, :] = async_data[m].get()

        probe = probes.pop()
        measurements = measurements.flatten()

    return measurements, probe


def sinogram(sx, sy, phantom, noise=False, pool=None):
    """Return a sinogram of phantom and the probe.

    Parameters
    ----------
    sx : int
        Number of rotation angles.
    sy : int
        Number of detection pixels (or sample translations).
    phantom : Phantom

    Returns
    -------
    sino : ndarray
        Sinogram.
    probe : Probe
        Probe with history.
    """
    scan = raster_scan(sx, sy)
    sino = np.zeros(sx*sy)

    sino, probe = calculate_gram(scan, sino, phantom, noise, pool,
                                 chunksize=sy)

    sino.shape = (sx, sy)

    return sino, probe


def angleogram(sx, sy, phantom, noise=False):
    """Return a angleogram of phantom and the probe.

    Parameters
    ----------
    sx : int
        Number of rotation angles.
    sy : int
        Number of detection pixels (or sample translations).
    phantom : Phantom

    Returns
    -------
    angl : ndarray
        Angleogram.
    probe : Probe
        Probe with history.
    """
    scan = angle_scan(sx, sy)
    angl = np.zeros(sx*sy)

    angl, probe = calculate_gram(scan, angl,  phantom, noise, pool,
                                 chunksize=sy)

    angl.shape = (sx, sy)

    return angl, probe


def raster_scan(sx, sy):
    """Provides a beam list for raster-scanning.

    The same Probe is returned each time to prevent recomputation of cached
    properties.

    Parameters
    ----------
    sx : int
        Number of rotation angles.
    sy : int
        Number of detection pixels (or sample translations).

    Yields
    ------
    Probe
    """
    # Step size of the probe.
    step = 1. / sy

    # Step size of the rotation angle.
    theta = np.pi / sx

    # Fixed probe location.
    p = Probe(Point([step / 2., -10]), Point([step / 2., 10]), step)

    for m in range(sx):
        for n in range(sy):
            yield p
            p.translate(step)
        p.translate(-1)
        p.rotate(theta, Point([0.5, 0.5]))


def angle_scan(sx, sy):
    """Provides a beam list for angle-scanning.

    The same Probe is returned each time to prevent recomputation of cached
    properties.

    Parameters
    ----------
    sx : int
        Number of rotation angles.
    sy : int
        Number of detection pixels (or sample translations).

    Yields
    ------
    Probe
    """
    # Step size of the probe.
    step = 0.1 / sy

    # Fixed rotation points.
    p1 = Point([0, 0.5])
    p2 = Point([0.5, 0.5])

    # Step size of the rotation angle.
    beta = np.pi / (sx + 1)
    alpha = np.pi / sy

    # Fixed probe location.
    p = Probe(Point([step / 2., -10]), Point([step / 2., 10]), step)

    for m in range(sx):
        for n in range(sy):
            yield p
            p.rotate(-alpha, p1)
        p.rotate(np.pi, p1)
        p1.rotate(-beta, p2)
        p.rotate(-beta, p2)
