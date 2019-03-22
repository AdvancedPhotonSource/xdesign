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
"""Defines methods for reconstructing data from the :mod:`.acquisition` module.

The algorithm module contains methods for reconstructing tomographic data
including gridrec, SIRT, ART, and MLEM. These methods can be used as benchmarks
for custom reconstruction methods or as an easy way to access reconstruction
algorithms for developing other methods such as noise correction.

.. note::
    Using `tomopy <https://github.com/tomopy/tomopy>`_ is recommended instead
    of these functions for heavy computation.

.. moduleauthor:: Doga Gursoy <dgursoy@aps.anl.gov>
"""

import logging

import numpy as np

from xdesign.acquisition import thv_to_zxy

logger = logging.getLogger(__name__)

__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['art', 'sirt', 'mlem', 'update_progress']


def update_progress(progress):
    """Draw a process bar in the terminal.

    Parameters
    -------------
    process : float
        The percentage completed e.g. 0.10 for 10%

    """
    percent = progress * 100
    nbars = int(progress * 10)
    print(
        '\r[{0}{1}] {2:.2f}%'.format('#' * nbars, ' ' * (10 - nbars), percent),
        end=''
    )
    if progress == 1:
        print('')


def get_mids_and_lengths(x0, y0, x1, y1, gx, gy):
    """Return the midpoints and intersection lengths of a line and a grid.

    Parameters
    ----------
    x0,y0,x1,y1 : float
        Two points which define the line. Points must be outside the grid
    gx,gy : :py:class:`np.array`
        Defines positions for the gridlines

    Return
    ------
    xm,ym : :py:class:`np.array`
        Coordinates along the line within each intersected grid pixel.
    dist : :py:class:`np.array`
        Lengths of the line segments crossing each pixel

    """
    # avoid upper-right boundary errors
    if (x1 - x0) == 0:
        x0 += 1e-6
    if (y1 - y0) == 0:
        y0 += 1e-6

    # vector lengths (ax, ay)
    ax = (gx - x0) / (x1 - x0)
    ay = (gy - y0) / (y1 - y0)

    # edges of alpha (a0, a1)
    ax0 = min(ax[0], ax[-1])
    ax1 = max(ax[0], ax[-1])
    ay0 = min(ay[0], ay[-1])
    ay1 = max(ay[0], ay[-1])
    a0 = max(max(ax0, ay0), 0)
    a1 = min(min(ax1, ay1), 1)

    # sorted alpha vector
    cx = (ax >= a0) & (ax <= a1)
    cy = (ay >= a0) & (ay <= a1)
    alpha = np.sort(np.r_[ax[cx], ay[cy]])

    # lengths
    xv = x0 + alpha * (x1 - x0)
    yv = y0 + alpha * (y1 - y0)
    lx = np.ediff1d(xv)
    ly = np.ediff1d(yv)
    dist = np.sqrt(lx**2 + ly**2)

    # indexing
    mid = alpha[:-1] + np.ediff1d(alpha) / 2.
    xm = x0 + mid * (x1 - x0)
    ym = y0 + mid * (y1 - y0)

    return xm, ym, dist


def art(
    gmin,
    gsize,
    data,
    theta,
    h,
    init,
    niter=10,
    weights=None,
    save_interval=None
):
    """Reconstruct data using ART algorithm. :cite:`Gordon1970`."""
    assert data.size == theta.size == h.size, "theta, h, must be" \
        "the equal lengths"
    data = data.ravel()
    theta = theta.ravel()
    h = h.ravel()
    if weights is None:
        weights = np.ones(data.shape)
    if save_interval is None:
        save_interval = niter
    archive = list()
    # Convert from probe to global coords
    srcx, srcy, detx, dety = thv_to_zxy(theta, h)
    # grid frame (gx, gy)
    sx, sy = init.shape
    gx = np.linspace(gmin[0], gmin[0] + gsize[0], sx + 1, endpoint=True)
    gy = np.linspace(gmin[1], gmin[1] + gsize[1], sy + 1, endpoint=True)
    midlengths = dict()  # cache the result of get_mids_and_lengths

    for n in range(niter):
        if n % save_interval == 0:
            archive.append(init.copy())

        # update = np.zeros(init.shape)
        # nupdate = np.zeros(init.shape, dtype=np.uint)

        update_progress(n / niter)
        for m in range(data.size):
            # get intersection locations and lengths
            if m in midlengths:
                xm, ym, dist = midlengths[m]
            else:
                xm, ym, dist = get_mids_and_lengths(
                    srcx[m], srcy[m], detx[m], dety[m], gx, gy
                )
                midlengths[m] = (xm, ym, dist)
            # convert midpoints of line segments to indices
            ix = np.floor(sx * (xm - gmin[0]) / gsize[0]).astype('int')
            iy = np.floor(sy * (ym - gmin[1]) / gsize[1]).astype('int')
            # simulate acquistion from initial guess
            dist2 = np.dot(dist, dist)
            if dist2 != 0:
                ind = (dist != 0) & (0 <= ix) & (ix < sx) \
                    & (0 <= iy) & (iy < sy)
                sim = np.dot(dist[ind], init[ix[ind], iy[ind]])
                upd = np.true_divide((data[m] - sim), dist2)
                init[ix[ind], iy[ind]] += dist[ind] * upd

    archive.append(init.copy())
    update_progress(1)
    if save_interval == niter:
        return init
    else:
        return archive


def sirt(
    gmin,
    gsize,
    data,
    theta,
    h,
    init,
    niter=10,
    weights=None,
    save_interval=None
):
    """Reconstruct data using SIRT algorithm. :cite:`Gilbert1972`."""
    assert data.size == theta.size == h.size, "theta, h, must be" \
        "the equal lengths"
    data = data.ravel()
    theta = theta.ravel()
    h = h.ravel()
    if weights is None:
        weights = np.ones(data.shape)
    if save_interval is None:
        save_interval = niter
    archive = list()
    # Convert from probe to global coords
    srcx, srcy, detx, dety = thv_to_zxy(theta, h)
    # grid frame (gx, gy)
    sx, sy = init.shape
    gx = np.linspace(gmin[0], gmin[0] + gsize[0], sx + 1, endpoint=True)
    gy = np.linspace(gmin[1], gmin[1] + gsize[1], sy + 1, endpoint=True)
    midlengths = dict()  # cache the result of get_mids_and_lengths

    for n in range(niter):
        if n % save_interval == 0:
            archive.append(init.copy())

        update = np.zeros(init.shape)
        nupdate = np.zeros(init.shape, dtype=np.uint)

        update_progress(n / niter)
        for m in range(data.size):
            # get intersection locations and lengths
            if m in midlengths:
                xm, ym, dist = midlengths[m]
            else:
                xm, ym, dist = get_mids_and_lengths(
                    srcx[m], srcy[m], detx[m], dety[m], gx, gy
                )
                midlengths[m] = (xm, ym, dist)
            # convert midpoints of line segments to indices
            ix = np.floor(sx * (xm - gmin[0]) / gsize[0]).astype('int')
            iy = np.floor(sy * (ym - gmin[1]) / gsize[1]).astype('int')
            # simulate acquistion from initial guess
            dist2 = np.dot(dist, dist)
            if dist2 != 0:
                ind = (dist != 0) & (0 <= ix) & (ix < sx) \
                    & (0 <= iy) & (iy < sy)
                sim = np.dot(dist[ind], init[ix[ind], iy[ind]])
                upd = np.true_divide((data[m] - sim), dist2)
                update[ix[ind], iy[ind]] += dist[ind] * upd
                nupdate[ix[ind], iy[ind]] += 1

        nupdate[nupdate == 0] = 1
        init += np.true_divide(update, nupdate)

    archive.append(init.copy())
    update_progress(1)
    if save_interval == niter:
        return init
    else:
        return archive


def mlem(gmin, gsize, data, theta, h, init, niter=10):
    """Reconstruct data using MLEM algorithm."""
    assert data.size == theta.size == h.size, "theta, h, must be" \
        "the equal lengths"
    data = data.ravel()
    theta = theta.ravel()
    h = h.ravel()
    # if weights is None:
    #     weights = np.ones(data.shape)
    # if save_interval is None:
    #     save_interval = niter
    # archive = list()
    # Convert from probe to global coords
    srcx, srcy, detx, dety = thv_to_zxy(theta, h)
    # grid frame (gx, gy)
    sx, sy = init.shape
    gx = np.linspace(gmin[0], gmin[0] + gsize[0], sx + 1, endpoint=True)
    gy = np.linspace(gmin[1], gmin[1] + gsize[1], sy + 1, endpoint=True)
    midlengths = dict()  # cache the result of get_mids_and_lengths

    for n in range(niter):

        update = np.zeros(init.shape)
        sumdist = np.zeros(init.shape)

        update_progress(n / niter)
        for m in range(data.size):
            # get intersection locations and lengths
            if m in midlengths:
                xm, ym, dist = midlengths[m]
            else:
                xm, ym, dist = get_mids_and_lengths(
                    srcx[m], srcy[m], detx[m], dety[m], gx, gy
                )
                midlengths[m] = (xm, ym, dist)
            # convert midpoints of line segments to indices
            ix = np.floor(sx * (xm - gmin[0]) / gsize[0]).astype('int')
            iy = np.floor(sy * (ym - gmin[1]) / gsize[1]).astype('int')
            # simulate acquistion from initial guess
            ind = (dist != 0)
            sumdist[ix[ind], iy[ind]] += dist
            sim = np.dot(dist[ind], init[ix[ind], iy[ind]])

            if not sim == 0:
                upd = np.true_divide(data[m], sim)
                update[ix[ind], iy[ind]] += dist[ind] * upd

        init[sumdist > 0] *= np.true_divide(
            update[sumdist > 0], sumdist[sumdist > 0] * sy
        )
    update_progress(1)
    return init
