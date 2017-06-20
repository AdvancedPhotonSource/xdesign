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
from copy import copy
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
    size : scalar, optional
        Size of the beam. i.e. the diameter
    """
    # TODO: Determine whether separate Beam object is necessary or if Beam can
    # be merged with Probe.
    def __init__(self, p1, p2, size=0):
        """Return a new Beam from two given points and optional size."""
        super(Beam, self).__init__(p1, p2)
        self.size = float(size)
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

    return beam.half_space.intersect(mesh.half_space).volume


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
    """
    # TODO: Implement additional attributes for Probe such as beam energy,
    # brightness, wavelength, etc.
    def __init__(self, p1, p2, size=0):
        super(Probe, self).__init__(p1, p2, size)
        self.history = []

    def __repr__(self):
        return "Probe({}, {}, size={})".format(repr(self.p1), repr(self.p2),
                                               repr(self.size))

    def translate(self, dx):
        """Translate beam along its normal direction."""
        vec = self.normal * dx
        super(Probe, self).translate(vec._x)

    def measure(self, phantom, sigma=0):
        """Return the probe measurement with optional Gaussian noise.

        Parameters
        ----------
        sigma : float >= 0
            The standard deviation of the normally distributed noise.
        """
        newdata = self._measure_helper(phantom)
        if sigma > 0:
            newdata += newdata * np.random.normal(scale=sigma)

        self.record()
        return newdata

    def _measure_helper(self, phantom):
        intersection = beamintersect(self, phantom.geometry)

        if intersection is not None and phantom.mass_atten != 0:
            newdata = intersection * phantom.mass_atten
        else:
            newdata = 0

        if intersection > 0:
            for child in phantom.children:
                newdata += self._measure_helper(child)

        return newdata

    def record(self):
        self.history.append(self.list)


def sinogram(sx, sy, phantom, noise=False):
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
    sino = np.zeros((sx, sy))
    for m in range(sx):
        for n in range(sy):
            probe = next(scan)
            sino[m, n] = probe.measure(phantom, noise)

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
    angl = np.zeros((sx, sy))
    for m in range(sx):
        for n in range(sy):
            probe = next(scan)
            angl[m, n] = probe.measure(phantom, noise)

    return angl, probe


def raster_scan(sa, st, width_fraction=1, nmeta=1, random=False, plot=False):
    """A :py:class:`.Probe` iterator for raster scanning.

    By default, the Probe position is the center of each translation step and
    there is no gap between steps. `width_fraction` < 1 creates a gap between
    Probes; `width_fraction` > 1 causes Probes to overlap. With `nmeta` > 1,
    the starting Probe position will offset after each rotation. When `nmeta`
    < 1, then `nmeta` will be auto calculately to completely cover the space.

    Parameters
    ----------
    sa : int
        The number of projeciton angles in `[0, 2PI)`.
    st : int
        The number of Probe steps at each projection angle.
    width_fraction : float
        The width of the Probe as a fraction of the step size (`1 / st`).
    nmeta : int >= 0
        The number of meta steps. Meta steps are the offset from the starting
        Probe position after each rotation.
    random : bool
        Whether the meta steps are organized in a random fashion or not.
    plot : bool
        Plot a angle vs position plot of the Probe positions.

    Yields
    ------
    p : :class:`.Probe`
    """
    step = 1. / st
    width = step * width_fraction

    if nmeta < 1:
        nmeta = int(np.ceil(step / width))
    meta_step = np.linspace(0, step, nmeta, endpoint=False)
    offset = step / nmeta / 2

    theta = 2 * np.pi / sa

    p = Probe(Point([offset, -10]), Point([offset, 10]), width)

    angles = list()
    positions = list()
    for m in range(sa):
        if random:
            np.random.shuffle(meta_step)
        p.translate(meta_step[m % nmeta])

        for n in range(st):
            angles.append(m * theta / np.pi)
            positions.append(offset + meta_step[m % nmeta] + n*step)
            yield p
            p.translate(step)

        p.translate(-1 - meta_step[m % nmeta])

        p.rotate(theta, Point([0.5, 0.5]))

    if plot:
        import matplotlib.pyplot as plt
        axis = plt.gca()
        axis.scatter(positions, angles)
        plt.xlabel('position [cm]')
        plt.ylabel('angle [pi rad]')
        plt.title('raster_scan({}, {}, width_fraction={},\nnmeta={},'
                  ' random={})'.format(sa, st, width_fraction, nmeta, random))


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
