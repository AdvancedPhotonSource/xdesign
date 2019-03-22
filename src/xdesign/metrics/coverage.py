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
"""Objects and methods for computing coverage based quality metrics.

These methods are based on the scanning trajectory only.

.. moduleauthor:: Daniel J Ching <carterbox@users.noreply.github.com>
"""

__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['coverage_approx']

import logging

from xdesign.acquisition import beamintersect, thv_to_zxy
from xdesign.recon import get_mids_and_lengths

logger = logging.getLogger(__name__)


def tensor_at_angle(angle, magnitude):
    """Return 2D tensor(s) with magnitude(s) at the angle [rad]."""
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    tensor = np.array([[1, 0], [0, 0]])
    tensor = np.einsum('...,jk->...jk', magnitude, tensor)
    return np.einsum('ij,...jk,lk->...il', R, tensor, R)


def coverage_approx(
    gmin,
    gsize,
    ngrid,
    probe_size,
    theta,
    h,
    v,
    weights=None,
    anisotropy=1,
    num_rays=16
):
    """Approximate procedure coverage with a Riemann sum.

    The intersection between the beam and each pixel is approximated by using a
    Reimann sum of `n` rectangles: width `beam.size / n` and length `dist`
    where `dist` is the length of segment of the line `alpha` which passes
    through the pixel parallel to the beam.

    If `anisotropy` is `True`, then `coverage_map.shape` is `(M, N, 2, 2)`,
    where the two extra dimensions contain coverage anisotopy information as a
    second order tensor.

    Parameters
    ----------
    procedure : :py:class:`.Probe` generator
        A generator which defines a scanning procedure by returning a sequence
        of Probe objects.
    region : :py:class:`np.array` [cm]
        A rectangle in which to map the coverage. Specify the bounds as
        `[[min_x, max_x], [min_y, max_y]]`. i.e. column vectors pointing to the
        min and max corner.
    pixel_size : float [cm]
        The edge length of the pixels in the coverage map in centimeters.
    n : int
        The number of lines per beam
    anisotropy : bool
        Whether the coverage map includes anisotropy information

    Returns
    -------
    coverage_map : :py:class:`numpy.ndarray`
        A discretized map of the Probe coverage.

    See also
    --------
    :py:func:`.plot.plot_coverage_anisotropy`
    """
    if weights is None:
        weights = np.ones(theta.shape)
    assert weights.size == theta.size == h.size == v.size, "theta, h, v must be" \
        "the equal lengths"
    coverage_map = np.zeros(list(ngrid) + [anisotropy])
    # split the probe up into bunches of rays
    line_offsets = np.linspace(0, probe_size, num_rays) - probe_size / 2
    theta = np.repeat(theta.flatten(), line_offsets.size)
    h = h.reshape(h.size, 1) + line_offsets
    h = h.flatten()
    v = np.repeat(v.flatten(), line_offsets.size)
    weights = np.repeat(weights.flatten(), line_offsets.size)
    # Convert from theta,h,v to x,y,z
    srcx, srcy, detx, dety, z = thv_to_zxy(theta, h, v)
    # grid frame (gx, gy)
    sx, sy = ngrid[0], ngrid[1]
    gx = np.linspace(gmin[0], gmin[0] + gsize[0], sx + 1, endpoint=True)
    gy = np.linspace(gmin[1], gmin[1] + gsize[1], sy + 1, endpoint=True)

    for m in range(theta.size):
        # get intersection locations and lengths
        xm, ym, dist = get_mids_and_lengths(
            srcx[m], srcy[m], detx[m], dety[m], gx, gy
        )
        if np.any(dist > 0):
            # convert midpoints of line segments to indices
            ix = np.floor(sx * (xm - gmin[0]) / gsize[0]).astype('int')
            iy = np.floor(sy * (ym - gmin[1]) / gsize[1]).astype('int')
            ia = np.floor((theta[m] / (np.pi / anisotropy) % anisotropy)
                          ).astype('int')
            ind = (dist != 0) & (0 <= ix) & (ix < sx) \
                & (0 <= iy) & (iy < sy)
            # put the weights in the binn
            coverage_map[ix[ind], iy[ind], ia] += dist[ind] * weights[m]

    pixel_area = np.prod(gsize) / np.prod(ngrid)
    line_width = probe_size / num_rays
    return coverage_map * line_width / pixel_area
