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
from xdesign.geometry import *
from xdesign.geometry import beamcirc, rotate
from xdesign.grid import *
from xdesign.propagation import *
from xdesign.algorithms import *
import dxchange
import logging
from itertools import izip
import h5py

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Beam',
           'Probe',
           'sinogram',
           'angleogram']


class Beam(Line):

    """Beam (thick line) in 2-D cartesian space.

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


class Probe(Beam):

    def __init__(self, p1, p2, size=0):
        super(Probe, self).__init__(p1, p2, size)
        self.history = []

    def translate(self, dx):
        """Translates beam along its normal direction."""
        vec = self.normal * dx
        self.p1 += vec
        self.p2 += vec

    def rotate(self, theta, origin):
        """Rotates beam around a given point."""
        self.p1 = rotate(self.p1, theta, origin)
        self.p2 = rotate(self.p2, theta, origin)

    def measure(self, phantom, noise=False):
        """Return the probe measurement given phantom. When noise is > 0,
        poisson noise is added to the returned measurement."""
        newdata = 0
        for m in range(phantom.population):
            newdata += (beamcirc(self, phantom.feature[m]) *
                        phantom.feature[m].mass_atten)
        if noise > 0:
            newdata += newdata * noise * np.random.poisson(1)
        self.record()
        return newdata

    def record(self):
        self.history.append(self.list)


def sinogram(sx, sy, phantom, noise=False):
    """Generates sinogram given a phantom.

    Parameters
    ----------
    sx : int
        Number of rotation angles.
    sy : int
        Number of detection pixels (or sample translations).
    phantom : Phantom

    Returns
    -------
    ndarray
        Sinogram.
    """
    scan = raster_scan(sx, sy)
    sino = np.zeros((sx, sy))
    for m in range(sx):
        # print(m)
        for n in range(sy):
            sino[m, n] = next(scan).measure(phantom, noise)
    return sino


def angleogram(sx, sy, phantom, noise=False):
    """Generates angleogram given a phantom.

    Parameters
    ----------
    sx : int
        Number of rotation angles.
    sy : int
        Number of detection pixels (or sample translations).
    phantom : Phantom

    Returns
    -------
    ndarray
        Angleogram.
    """
    scan = angle_scan(sx, sy)
    angl = np.zeros((sx, sy))
    for m in range(sx):
        print(m)
        for n in range(sy):
            angl[m, n] = next(scan).measure(phantom, noise)
    return angl


def raster_scan(sx, sy):
    """Provides a beam list for raster-scanning.

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
    p = Probe(Point(step / 2., -10), Point(step / 2., 10), step)

    for m in range(sx):
        for n in range(sy):
            yield p
            p.translate(step)
        p.translate(-1)
        p.rotate(theta, Point(0.5, 0.5))


def angle_scan(sx, sy):
    """Provides a beam list for raster-scanning.

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
    p1 = Point(0, 0.5)
    p2 = Point(0.5, 0.5)

    # Step size of the rotation angle.
    beta = np.pi / (sx + 1)
    alpha = np.pi / sy

    # Fixed probe location.
    p = Probe(Point(step / 2., -10), Point(step / 2., 10), step)

    for m in range(sx):
        for n in range(sy):
            yield p
            p.rotate(-alpha, p1)
        p.rotate(np.pi, p1)
        p1.rotate(-beta, p2)
        p.rotate(-beta, p2)


def tomography_3d(grid, wavefront, probe, ang_start, ang_end, ang_step=None, n_ang=None, savefolder='tomo_output', fname='tomo', format='h5', pr=None, **kwargs):
    allowed_kwargs = {'mba': ['alpha']}
    if pr not in allowed_kwargs:
        raise ValueError
    for key, value in list(kwargs.items()):
        if key not in allowed_kwargs[pr]:
            raise ValueError('Invalid options for selected phase retrieval method.')
        else:
            if pr == 'mba':
                alpha = kwargs['alpha']
                print(alpha)
    assert isinstance(grid, Grid3d)
    if not ang_step is None:
        angles = np.arange(ang_start, ang_end + ang_step, ang_step)
        n_ang = int((ang_end - ang_start) / ang_step) + 1
    elif not n_ang is None:
        angles = np.linspace(ang_start, ang_end, n_ang)
        ang_step = float(ang_end - ang_start) / (n_ang - 1)
    else:
        print('ERROR:xdesign.acquisition:Angular step or number of angles should be specified.')
        return
    if format == 'h5':
        f = h5py.File('{:s}/{:s}.h5'.format(savefolder, fname))
        xchng = f.create_group('exchange')
        dset = xchng.create_dataset('data', (n_ang, grid.size[1], grid.size[2]), dtype='float32')
    for theta, i in izip(angles, range(n_ang)):
        print('\rNow at angle ', str(theta), end='')
        exiting = multislice_propagate(grid, probe, wavefront)
        exiting = np.real(exiting * np.conjugate(exiting))
        if not pr is None:
            if pr == 'mba':
                exiting = mba(exiting, (grid.voxel_y, grid.voxel_x), alpha=alpha)
        if format == 'tiff':
            dxchange.write_tiff(exiting, fname='{:s}/{:s}_{:05}.tiff'.format(savefolder, fname, i), dtype='float32')
        else:
            dset[i, :, :] = exiting
        grid.rotate(ang_step)
    if format == 'h5':
        dset = xchng.create_dataset('data_white', (1, grid.size[1], grid.size[2]))
        dset[:, :, :] = np.ones(dset.shape)
        dset = xchng.create_dataset('data_dark', (1, grid.size[1], grid.size[2]))
        dset[:, :, :] = np.zeros(dset.shape)
        f.close()