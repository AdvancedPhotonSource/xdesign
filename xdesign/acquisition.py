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
.. moduleauthor:: Daniel J Ching <carterbox@users.noreply.github.com>
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
try:
    import queue
except ImportError:
    import Queue as queue
import time

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Probe',
           'calculate_gram',
           'sinogram',
           'raster_scan',
           'raster_scan3D']


class Probe(Line, pt.Polytope):
    """An x-ray beam for probing Phantoms.

    A Probe is initialized by two points and a size (diameter).

    Attributes
    ----------
    p1 : Point
    p2 : Point
    size : float, cm (default: 0.0 cm)
        The size of probe in centimeters.
    intensity : float, cd (default: 1.0 cd)
        The intensity of the beam in candela.
    energy : float, eV (defauly: 15 eV)
        The energy of the probe in eV.


    .. todo::
        Implement additional attributes for Probe such as wavelength,
        etc.

    """
    def __init__(self, p1, p2, size=0.0, intensity=1.0, energy=15.0,
                 circleapprox=32):
        super(Probe, self).__init__(p1, p2)

        self.size = size
        self.intensity = intensity
        self.energy = energy

        self.history = list()

        # Construct the Polytope beam
        # determine the length, position, and shape of the beam
        radius = self.size / 2
        half_length = self.p1.distance(self.p2) / 2
        center = ((self.p2 + self.p1) / 2)._x

        # make a bounding box around axis 0
        hi = np.full(self.dim, radius)
        lo = -hi
        hi[0] = +half_length
        lo[0] = -half_length

        # create the polytope
        p = pt.Polytope.from_box(np.stack([lo, hi], axis=1))

        # Rotate the polytope around axis 0 to create a cylinder
        if self.dim > 2:
            nrotations = circleapprox
            angle = np.pi / (2 * (nrotations + 1))
            for i in range(nrotations):
                rotated = p.rotation(i=1, j=2, theta=angle)
                p = pt.intersect(p, rotated)

        # find the vector which bisects the angle between [1,0,0] and the beam
        u = Point([1] + [0]*(self.dim - 1))
        v = self.p2 - self.p1
        w = u.norm * v._x + v.norm * u._x

        # rotate the polytope and translate to beam
        pt.polytope._rotate(p, u=u._x, v=w)
        p = p.translation(center)

        pt.Polytope.__init__(self, A=p.A, b=p.b, minrep=True)

    def __repr__(self):
        return "Probe({}, {}, size={}, intensity={}, energy={})".format(
                repr(self.p1), repr(self.p2), repr(self.size),
                repr(self.intensity), repr(self.energy))

    def __str__(self):
        """Return the string respresentation of the Beam."""
        return "Probe(" + super(Probe, self).__str__() + ")"

    def translate(self, vector):
        """Translate entity along vector."""
        logger.debug("Probe.translate: {}".format(vector))

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        if vector.size < 2:
            vec = self.normal * np.asscalar(vector)
            vector = vec._x

        self.p1.translate(vector)
        self.p2.translate(vector)
        pt.polytope._translate(self, vector)

    def record(self):
        self.history.append(self.list)

    def distance(self, other):
        """Return the closest distance between entities."""
        dx = super(Probe, self).distance(other)
        return dx - self.size / 2

    def rotate(self, theta, point=None, axis=None):
        """Rotate entity around an axis which passes through an point by theta
        radians."""
        logger.debug("Probe.rotate: {}, {}, {}".format(theta, point, axis))
        self.p1.rotate(theta, point, axis)
        self.p2.rotate(theta, point, axis)

        if point is None:
            d = 0
        else:
            d = point._x

        pt.polytope._translate(self, -d)
        pt.polytope._rotate(self, i=0, j=1, theta=theta)
        pt.polytope._translate(self, d)

    def measure(self, phantom, perc=0.0, pool=None):
        """Return the probe measurement with optional Gaussian noise.

        Parameters
        ----------
        perc : float >= 0
            Percentage of Gaussian noise.
        """
        newdata = self.intensity * np.exp(-self._get_attenuation(phantom))

        if perc > 0:
            newdata += newdata * perc * np.random.normal(scale=1)

        self.record()

        logger.debug("Probe.measure: {}".format(newdata))
        return newdata

    def _get_attenuation(self, phantom):
        """Return the beam intensity attenuation due to the phantom."""
        intersection = beamintersect(self, phantom.geometry)

        if intersection is None or phantom.material is None:
            attenuation = 0.0
        else:
            # [ ] = [cm^2] / [cm] * [1/cm]
            attenuation = (intersection / self.cross_section
                           * phantom.material.linear_attenuation(self.energy))

        if phantom.geometry is None or intersection > 0:
            # check the children for containers and intersecting geometries
            for child in phantom.children:
                attenuation += self._get_attenuation(child)

        return attenuation

    @cached_property
    def cross_section(self):
        if self.dim == 2:
            return self.size
        else:
            return np.pi * self.size**2 / 4


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
    elif isinstance(geometry, pt.Polytope):
        return beamtope(beam, geometry)
    else:
        raise NotImplementedError


def beammesh(beam, mesh):
    """Intersection area of infinite beam with polygonal mesh"""
    if beam.distance(mesh.center) > mesh.radius:
        logger.debug("BEAMMESH: skipped because of radius.")
        return 0

    volume = 0

    for f in mesh.faces:
        volume += f.sign * beamintersect(beam, f)

    return volume


def beampoly(beam, poly):
    """Intersection area of an infinite beam with a polygon"""
    if beam.distance(poly.center) > poly.radius:
        logger.debug("BEAMPOLY: skipped because of radius.")
        return 0

    return beam.intersect(poly.half_space).volume


def beamtope(beam, tope):
    """Intersection area of an infinite beam with a polytope"""
    # if beam.distance(Point(tope.chebXc)) > tope.radius:
    #     logger.debug("BEAMTOPE: skipped because of radius.")
    #     return 0

    return beam.intersect(tope).volume


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
    p = super(Probe, beam).distance(circle.center)
    assert(p >= 0)

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
    assert(a >= 0), a
    return a


def probe_wrapper(probes, phantom, **kwargs):
    """Wrap probe.measure to make it suitable for multiprocessing.

    This method does two things: (1) it puts the Probe.measure method in the
    f(*args) format for the pool workers (2) it passes chunks of work (multiple
    probes) to the workers to reduce overhead.
    """
    measurements = [None] * len(probes)
    for i in range(len(probes)):
        measurements[i] = probes[i].measure(phantom, **kwargs)

    return measurements


def calculate_gram(procedure, niter, phantom, pool=None,
                   chunksize=1, mkwargs={}):
    """Measure the `phantom` using the `procedure`.

    Parameters
    ----------
    procedure : :py:`.iterator`
        An iterator that yields :class:`.Probe`
    niter : int
        The number of measurements to take
    phantom : :class:`.phantom.Phantom`
    pool : :py:`.multiprocessing.Pool` (default: None)
    chunksize : int (default: 1)
        The number of measurements to send to each worker in the pool.
    mkwargs : dict
        keyword arguments to pass to :ref:`.Probe.measure`.

    Raise
    -----
    ValueError
        If niter is not a multiple of chunksize

    .. seealso::
        :class:`.sinogram`, :class:`,angelogram`
    """
    measurements = np.zeros(niter)

    if pool is None:  # no multiprocessing
        logging.info("calculate_gram: {}, single "
                     "thread".format(procedure.__name__))

        for m in range(niter):
            probe = next(procedure)
            measurements[m] = probe.measure(phantom, **mkwargs)

    else:
        if chunksize == 1:
            warnings.warn("Default chunksize is 1. This is not optimal.",
                          RuntimeWarning)

        nchunks = niter // chunksize
        measurements.shape = (nchunks, chunksize)
        logging.info("calculate_gram: {}, {} "
                     "chunks".format(procedure.__name__, nchunks))

        # assign work to pool
        nmaxqueue = pool._processes * 2
        async_data = queue.Queue(nmaxqueue)
        for m in range(nchunks):
            while async_data.full():
                item = async_data.get()

                if item[1].ready():
                    if item[1].successful():
                        measurements[item[0], :] = item[1].get()
                    else:
                        raise RuntimeError('Process Failed '
                                           'at chunk {}'.format(item[0]))
                else:  # not ready
                    async_data.put(item)
                    time.sleep(5)

            probes = [None] * chunksize

            for n in range(chunksize):
                probe = next(procedure)
                probe.record()
                probe_copy = deepcopy(probe)
                probe_copy.history = []
                probes[n] = probe_copy

            async_data.put((m,
                            pool.apply_async(probe_wrapper,
                                             (probes, phantom),
                                             mkwargs))
                           )
            logging.info('calculate_gram: chunk {} queued'.format(m))

        while not async_data.empty():
            item = async_data.get()
            item[1].wait()
            if item[1].successful():
                measurements[item[0], :] = item[1].get()
            else:
                raise RuntimeError('Process Failed '
                                   'at chunk {}'.format(item[0]))

        measurements = measurements.flatten()

    return measurements, probe


def sinogram(sx, sy, phantom, pool=None, mkwargs={}):
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

    sino, probe = calculate_gram(scan, sx*sy, phantom,
                                 pool=pool, chunksize=sy, mkwargs=mkwargs)

    sino.shape = (sx, sy)

    return sino, probe


def raster_scan3D(sz, sa, st, zstart=None):
    """A Probe iterator for raster-scanning in 3D.

    The size of the probe is 1 / st.

    Parameters
    ----------
    sz : int
        The number of vertical slices.
    sa : int
        The number of rotation angles over PI/2
    st : int
        The number of detection pixels (or sample translations).

    Yields
    ------
    p : Probe
    """
    # Step sizes of the probe.
    tstep = Point([0, 1. / st, 0])
    zstep = Point([0, 0, 1. / sz])
    theta = np.pi / sa

    # Fixed probe location.
    if zstart is None:
        zstart = 1. / sz / 2.

    p = Probe(Point([-10, 1. / st / 2., zstart]),
              Point([10, 1. / st / 2., zstart]),
              size=1. / st)

    for o in range(sz):
        for m in range(sa):
            for n in range(st):
                yield p
                p.translate(tstep._x)
            p.translate(-st * tstep._x)
            p.rotate(theta, Point([0.5, 0.5, 0]))
            tstep.rotate(theta)
        p.rotate(np.pi, Point([0.5, 0.5, 0]))
        tstep.rotate(np.pi)
        p.translate(zstep._x)


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
