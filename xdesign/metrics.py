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
"""Objects and methods for computing the quality of reconstructions.

.. moduleauthor:: Daniel J Ching <carterbox@users.noreply.github.com>
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import phasepack as phase

from scipy import ndimage
from scipy import optimize
from scipy import stats
from copy import deepcopy

from xdesign.phantom import HyperbolicConcentric, UnitCircle
from xdesign.acquisition import beamintersect, thv_to_zxy
from xdesign.geometry import Circle, Point, Line
from xdesign.algorithms import get_mids_and_lengths

logger = logging.getLogger(__name__)

__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['compute_PCC',
           'compute_mtf',
           'compute_mtf_ffst',
           'compute_mtf_lwkj',
           'compute_nps_ffst',
           'compute_neq_d',
           'ImageQuality',
           'compute_ssim',
           'compute_msssim',
           'coverage_approx']


def tensor_at_angle(angle, magnitude):
    """Return 2D tensor(s) with magnitude(s) at the angle [rad]."""
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    tensor = np.array([[1, 0], [0, 0]])
    tensor = np.einsum('...,jk->...jk', magnitude, tensor)
    return np.einsum('ij,...jk,lk->...il', R, tensor, R)


def coverage_approx(gmin, gsize, ngrid, probe_size, theta, h, v, weights=None,
                    anisotropy=1, num_rays=16):
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
        xm, ym, dist = get_mids_and_lengths(srcx[m], srcy[m],
                                            detx[m], dety[m],
                                            gx, gy)
        if np.any(dist > 0):
            # convert midpoints of line segments to indices
            ix = np.floor(sx * (xm - gmin[0]) / gsize[0]).astype('int')
            iy = np.floor(sy * (ym - gmin[1]) / gsize[1]).astype('int')
            ia = np.floor((theta[m] / (np.pi / anisotropy)
                          % anisotropy)).astype('int')
            ind = (dist != 0) & (0 <= ix) & (ix < sx) \
                & (0 <= iy) & (iy < sy)
            # put the weights in the binn
            coverage_map[ix[ind], iy[ind], ia] += dist[ind] * weights[m]

    pixel_area = np.prod(gsize) / np.prod(ngrid)
    line_width = probe_size / num_rays
    return coverage_map * line_width / pixel_area


def compute_mtf(phantom, image):
    """Approximate the modulation tranfer function using the
    HyperbolicCocentric phantom. Calculate the MTF from the modulation depth
    at each edge on the line from (0.5,0.5) to (0.5,1). MTF = (hi-lo)/(hi+lo)

    Parameters
    ---------------
    phantom : HyperbolicConcentric
        Predefined phantom of cocentric rings whose widths decay parabolically.
    image : ndarray
        The reconstruction of the above phantom.

    Returns
    --------------
    wavelength : list
        wavelenth in the scale of the original phantom
    MTF : list
        MTF values

    .. deprecated:: 0.3
        This method rapidly becomes inaccurate at small wavelenths because the
        measurement gets out of phase with the waves due to rounding error. Use
        another one of the MTF functions instead.

    .. seealso::
        :meth:`compute_mtf_ffst`
        :meth:`compute_mtf_lwkj`
    """
    warnings.warn('compute_mtf is decprecated, use compute_mtf_lwkj or ' +
                  'compute_mtf_ffst instead', DeprecationWarning)

    if not isinstance(phantom, HyperbolicConcentric):
        raise TypeError

    center = int(image.shape[0] / 2)  # assume square shape
    radii = np.array(phantom.radii) * image.shape[0]
    widths = np.array(phantom.widths) * image.shape[0]

    # plt.figure()
    # plt.plot(image[int(center),:])
    # plt.show(block=True)

    MTF = []
    for i in range(1, len(widths) - 1):
        # Locate the edge between rings in the discrete reconstruction.
        mid = int(center + radii[i])  # middle of edge
        rig = int(mid + widths[i + 1])  # right boundary
        lef = int(mid - widths[i + 1])  # left boundary
        # print(lef,mid,rig)

        # Stop when the waves are below the size of a pixel
        if rig == mid or lef == mid:
            break

        # Calculate MTF at the edge
        hi = np.sum(image[center, lef:mid])
        lo = np.sum(image[center, mid:rig])
        MTF.append(abs(hi - lo) / (hi + lo))

        # plt.figure()
        # plt.plot(image[int(center),int(lef):int(rig)])
        # plt.show(block=True)

    wavelength = phantom.widths[1:-1]
    return wavelength, MTF


def compute_mtf_ffst(phantom, image, Ntheta=4):
    '''Calculate the MTF using the method described in :cite:`Friedman:13`.

    Parameters
    ----------
    phantom : :py:class:`.UnitCircle`
        Predefined phantom with single circle whose radius is less than 0.5.
    image : ndarray
        The reconstruction of the phantom.
    Ntheta : scalar
        The number of directions at which to calculate the MTF.

    Returns
    -------
    wavenumber : ndarray
        wavelenth in the scale of the original phantom
    MTF : ndarray
        MTF values
    bin_centers : ndarray
        the center of the bins if Ntheta >= 1

    .. seealso::
        :meth:`compute_mtf_lwkj`
    '''
    if not isinstance(phantom, UnitCircle):
        raise TypeError('MTF requires unit circle phantom.')
    if phantom.geometry.radius >= 0.5:
        raise ValueError('Radius of the phantom should be less than 0.5.')
    if Ntheta <= 0:
        raise ValueError('Must calculate MTF in at least one direction.')
    if not isinstance(image, np.ndarray):
        raise TypeError('image must be numpy.ndarray')

    # convert pixel coordinates to length coordinates
    x = y = (np.arange(0, image.shape[0]) / image.shape[0] - 0.5)
    X, Y = np.meshgrid(x, y)
    # calculate polar coordinates for each position
    R = np.sqrt(X**2 + Y**2)
    Th = np.arctan2(Y, X)
    # print(x)

    # Normalize the data to [0,1)
    x_circle = np.mean(image[R < phantom.geometry.radius - 0.01])
    x_air = np.mean(image[R > phantom.geometry.radius + 0.01])
    # print(x_air)
    # print(x_circle)
    image = (image - x_air) / (x_circle - x_air)
    image[image < 0] = 0
    image[image > 1] = 1

    # [length] (R is already converted to length)
    R_bin_width = 1 / image.shape[0]
    R_bins = np.arange(0, np.max(R), R_bin_width)
    # print(R_bins)

    Th_bin_width = 2 * np.pi / Ntheta  # [radians]
    Th_bins = np.arange(-Th_bin_width / 2, 2 * np.pi -
                        Th_bin_width / 2, Th_bin_width)
    Th[Th < -Th_bin_width / 2] = 2 * np.pi + Th[Th < -Th_bin_width / 2]
    # print(Th_bins)

    # data with radius falling within a given bin are averaged together for a
    # low noise approximation of the ESF at the given radius
    ESF = np.empty([Th_bins.size, R_bins.size])
    ESF[:] = np.NAN
    count = np.zeros([Th_bins.size, R_bins.size])
    for r in range(0, R_bins.size):
        Rmask = R_bins[r] <= R
        if r + 1 < R_bins.size:
            Rmask = np.bitwise_and(Rmask, R < R_bins[r + 1])

        for th in range(0, Th_bins.size):
            Tmask = Th_bins[th] <= Th
            if th + 1 < Th_bins.size:
                Tmask = np.bitwise_and(Tmask, Th < Th_bins[th + 1])

            # average all the counts for equal radii
            # TODO: Determine whether count is actually needed. It could be
            # replaced with np.mean
            mask = np.bitwise_and(Tmask, Rmask)
            count[th, r] = np.sum(mask)
            if 0 < count[th, r]:  # some bins may be empty
                ESF[th, r] = np.sum(image[mask]) / count[th, r]

    while np.sum(np.isnan(ESF)):  # smooth over empty bins
        ESF[np.isnan(ESF)] = ESF[np.roll(np.isnan(ESF), -1)]

    # plt.figure()
    # for i in range(0,ESF.shape[0]):
    #    plt.plot(ESF[i,:])
    # plt.xlabel('radius');
    # plt.title('ESF')

    LSF = -np.diff(ESF, axis=1)

    # trim the LSF so that the edge is in the center of the data
    edge_center = int(phantom.geometry.radius / R_bin_width)
    # print(edge_center)
    pad = int(LSF.shape[1] / 5)
    LSF = LSF[:, edge_center - pad:edge_center + pad + 1]
    # print(LSF)
    LSF_weighted = LSF * np.hanning(LSF.shape[1])

    # plt.figure()
    # for i in range(0,LSF.shape[0]):
    #    plt.plot(LSF[i,:])
    # plt.xlabel('radius');
    # plt.title('LSF')
    # plt.show(block=True)

    # Calculate the MTF
    T = np.fft.fftshift(np.fft.fft(LSF_weighted))
    faxis = (np.arange(0, LSF.shape[1]) / LSF.shape[1] - 0.5) / R_bin_width
    nyquist = 0.5*image.shape[0]

    MTF = np.abs(T)
    bin_centers = Th_bins + Th_bin_width/2

    return faxis, MTF, bin_centers


def compute_mtf_lwkj(phantom, image):
    """Calculate the MTF using the modulated Siemens Star method in
    :cite:`loebich2007digital`.

    Parameters
    ----------
    phantom : :py:class:`.SiemensStar`
    image : ndarray
        The reconstruciton of the SiemensStar

    Returns
    -------
    frequency : array
        The spatial frequency in cycles per unit length
    M : array
        The MTF values for each frequency

    .. seealso::
        :meth:`compute_mtf_ffst`
    """
    # Determine which radii to sample. Do not sample linearly because the
    # spatial frequency changes as 1/r
    Nradii = 100
    Nangles = 256
    pradii = 1/1.05**np.arange(1, Nradii)  # proportional radii of the star

    line, theta = get_line_at_radius(image, pradii, Nangles)
    M = fit_sinusoid(line, theta, phantom.n_sectors/2)

    # convert from contrast as a function of radius to contrast as a function
    # of spatial frequency
    frequency = phantom.ratio/pradii.flatten()

    return frequency, M


def get_line_at_radius(image, fradius, N):
    """Return an Nx1 array of the values of the image at a radius.

    Parameters
    ----------
    image : :py:class:`numpy.ndarray`
        A centered image of the seimens star.
    fradius : :py:class:`numpy.array_like`
        The M radius fractions of the image at which to extract the line
        given as a floats in the range (0, 1).
    N : int
        The number of points to sample around the circumference of each circle

    Returns
    -------
    line : NxM :py:class:`numpy.ndarray`
        the values from image at the radius
    theta : Nx1 :py:class:`numpy.ndarray`
        the angles that were sampled in radians

    Raises
    ------
    ValueError
        If `image` is not square.
        If any value of `fradius` is not in the range (0, 1).
        If `N` < 1.
    """
    fradius = np.asanyarray(fradius)
    if image.shape[0] != image.shape[1]:
        raise ValueError('image must be square.')
    if np.any(0 >= fradius) or np.any(fradius >= 1):
        raise ValueError('fradius must be in the range (0, 1)')
    if N < 1:
        raise ValueError('Sampling less than 1 point is not useful.')
    # add singleton dimension to enable matrix multiplication
    M = fradius.size
    fradius.shape = (1, M)
    # calculate the angles to sample
    theta = np.arange(0, N) / N * 2 * np.pi
    theta.shape = (N, 1)
    # convert the angles to xy coordinates
    x = fradius * np.cos(theta)
    y = fradius * np.sin(theta)
    # round to nearest integer location and shift to center
    image_half = image.shape[0]/2
    x = np.round((x + 1) * image_half)
    y = np.round((y + 1) * image_half)
    # extract from image
    line = image[x.astype(int), y.astype(int)]
    assert line.shape == (N, M), line.shape
    assert theta.shape == (N, 1), theta.shape
    return line, theta


def fit_sinusoid(value, angle, f, p0=[0.5, 0.25, 0.25]):
    """Fit a periodic function of known frequency, f, to the value and angle
    data. value = Func(angle, f). NOTE: Because the fiting function is
    sinusoidal instead of square, contrast values larger than unity are clipped
    back to unity.

    parameters
    ----------
    value : NxM ndarray
        The value of the function at N angles and M radii
    angle : Nx1 ndarray
        The N angles at which the function was sampled
    f : scalar
        The expected angular frequency; the number of black/white pairs in
        the siemens star. i.e. half the number of spokes
    p0 : list, optional
        The initial guesses for the parameters.

    returns:
    --------
    MTFR: 1xM ndarray
        The modulation part of the MTF at each of the M radii
    """
    M = value.shape[1]

    # Distance to the target function
    def errorfunc(p, x, y): return periodic_function(p, x) - y

    time = np.linspace(0, 2*np.pi, 100)

    MTFR = np.ndarray((1, M))
    x = (f*angle).squeeze()
    for radius in range(0, M):
        p1, success = optimize.leastsq(errorfunc, p0[:],
                                       args=(x, value[:, radius]))

        # print(success)
        # plt.figure()
        # plt.plot(angle, value[:, radius], "ro",
        #          time, periodic_function(p1, f*time), "r-")

        MTFR[:, radius] = np.sqrt(p1[1]**2 + p1[2]**2)/p1[0]

    # cap the MTF at unity
    MTFR[MTFR > 1.] = 1.
    assert(not np.any(MTFR < 0)), MTFR
    assert(MTFR.shape == (1, M)), MTFR.shape
    return MTFR


def periodic_function(p, x):
    """A periodic function for fitting to the spokes of the Siemens Star.

    parameters
    ----------
    p[0] : scalar
        the mean of the function
    p[1], p[2] : scalar
        the amplitudes of the function
    x : Nx1 ndarray
        the angular frequency multiplied by the angles for the function.
        w * theta
    w : scalar
        the angular frequency; the number of black/white pairs in the siemens
        star. i.e. half the number of spokes
    theta : Nx1 ndarray
        input angles for the function

    returns
    -------
    value : Nx1 array
        the values of the function at phi; cannot return NaNs.
    """
    # x = w * theta
    value = p[0] + p[1] * np.sin(x) + p[2] * np.cos(x)
    assert(value.shape == x.shape), (value.shape, theta.shape)
    assert(not np.any(np.isnan(value)))
    return value


def compute_nps_ffst(phantom, A, B=None, plot_type='frequency'):
    '''Calculate the noise power spectrum from a unit circle image using the
    method from :cite:`Friedman:13`.

    Parameters
    ----------
    phantom : UnitCircle
        The unit circle phantom.
    A : ndarray
        The reconstruction of the above phantom.
    B : ndarray
        The reconstruction of the above phantom with different noise. This
        second reconstruction enables allows use of trend subtraction instead
        of zero mean normalization.
    plot_type : string
        'histogram' returns a plot binned by radial coordinate wavenumber
        'frequency' returns a wavenumber vs wavenumber plot

    returns
    -------
    bins :
        Bins for the radially binned NPS
    counts :
        NPS values for the radially binned NPS
    X, Y :
        Frequencies for the 2D frequency plot NPS
    NPS : 2Darray
        the NPS for the 2D frequency plot
    '''
    if not isinstance(phantom, UnitCircle):
        raise TypeError('NPS requires unit circle phantom.')
    if not isinstance(A, np.ndarray):
        raise TypeError('A must be numpy.ndarray.')
    if not isinstance(B, np.ndarray):
        raise TypeError('B must be numpy.ndarray.')
    if A.shape != B.shape:
        raise ValueError('A and B must be the same size!')
    if not (plot_type == 'frequency' or plot_type == 'histogram'):
        raise ValueError("plot type must be 'frequency' or 'histogram'.")

    image = A
    if B is not None:
        image = image - B

    # plt.figure()
    # plt.imshow(image, cmap='inferno',interpolation="none")
    # plt.colorbar()
    # plt.show(block=True)

    resolution = image.shape[0]  # [pixels/length]
    # cut out uniform region (square circumscribed by unit circle)
    i_half = int(image.shape[0] / 2)  # half image
    # half of the square inside the circle
    s_half = int(image.shape[0] * phantom.geometry.radius / np.sqrt(2))
    unif_region = image[i_half - s_half:i_half +
                        s_half, i_half - s_half:i_half + s_half]

    # zero-mean normalization
    unif_region = unif_region - np.mean(unif_region)

    # 2D fourier-transform
    unif_region = np.fft.fftshift(np.fft.fft2(unif_region))
    # squared modulus / squared complex norm
    NPS = np.abs((unif_region))**2  # [attenuation^2]

    # Calculate axis labels
    # TODO@dgursoy is this frequency scaling correct?
    x = y = (np.arange(0, unif_region.shape[0]) /
             unif_region.shape[0] - 0.5) * image.shape[0]
    X, Y = np.meshgrid(x, y)
    # print(x)

    if plot_type == 'histogram':
        # calculate polar coordinates for each position
        R = np.sqrt(X**2 + Y**2)
        # Theta = nothing; we are averaging radial contours

        bin_width = 1  # [length] (R is already converted to length)
        bins = np.arange(0, np.max(R), bin_width)
        # print(bins)
        counts = np.zeros(bins.shape)
        for i in range(0, bins.size):
            if i < bins.size - 1:
                mask = np.bitwise_and(bins[i] <= R, R < bins[i + 1])
            else:
                mask = R >= bins[i]
            # average all the counts for equal radii
            if 0 < np.sum(mask):  # some bins may be empty
                counts[i] = np.mean(NPS[mask])

        return bins, counts

    elif plot_type == 'frequency':
        return X, Y, NPS


def compute_neq_d(phantom, A, B):
    '''Calculate the NEQ according to recommendations by :cite:`Dobbins:95`.

    Parameters
    ----------
    phantom : UnitCircle
        The unit circle class with radius less than 0.5
    A : ndarray
        The reconstruction of the above phantom.
    B : ndarray
        The reconstruction of the above phantom with different noise. This
        second reconstruction enables allows use of trend subtraction instead
        of zero mean normalization.

    Returns
    -------
    mu_b :
        The spatial frequencies
    NEQ :
        the Noise Equivalent Quanta
    '''
    mu_a, NPS = compute_nps_ffst(phantom, A, B, plot_type='histogram')
    mu_b, MTF, bins = compute_mtf_ffst(phantom, A, Ntheta=1)

    # remove negative MT
    MTF = MTF[:, mu_b > 0]
    mu_b = mu_b[mu_b > 0]

    # bin the NPS data to match the MTF data
    NPS_binned = np.zeros(MTF.shape)
    for i in range(0, mu_b.size):
        bucket = mu_b[i] < mu_a
        if i + 1 < mu_b.size:
            bucket = np.logical_and(bucket, mu_a < mu_b[i+1])

        if NPS[bucket].size > 0:
            NPS_binned[0, i] = np.sum(NPS[bucket])

    NEQ = MTF/np.sqrt(NPS_binned)  # or something similiar

    return mu_b, NEQ


def compute_PCC(A, B, masks=None):
    """Computes the Pearson product-moment correlation coefficients (PCC) for
    the two images.

    Parameters
    -------------
    A,B : ndarray
        The two images to be compared
    masks : list of ndarrays, optional
        If supplied, the data under each mask is computed separately.

    Returns
    ----------------
    covariances : array, list of arrays
    """
    covariances = []
    if masks is None:
        data = np.vstack((np.ravel(A), np.ravel(B)))
        return np.corrcoef(data)

    for m in masks:
        weights = m[m > 0]
        masked_B = B[m > 0]
        masked_A = A[m > 0]
        data = np.vstack((masked_A, masked_B))
        # covariances.append(np.cov(data,aweights=weights))
        covariances.append(np.corrcoef(data))

    return covariances


class ImageQuality(object):
    """Store information about image quality.

    Attributes
    ----------
    img0 : array
    img1 : array
        Stacks of reference and deformed images.
    metrics : dict
        A dictionary with image quality information organized by scale.
        ``metric[scale] = (mean_quality, quality_map)``
    method : string
        The metric used to calculate the quality
    """

    def __init__(self, original, reconstruction):
        self.img0 = original.astype(np.float)
        self.img1 = reconstruction.astype(np.float)

        self.scales = None
        self.mets = None
        self.maps = None
        self.method = ''

    def compute_quality(self, method="MSSSIM", L=1.0, **kwargs):
        """Compute the full-reference image quality of each image pair.

        Available methods include SSIM :cite:`wang:02`, MSSSIM :cite:`wang:03`,
        VIFp :cite:`Sheikh:15`, and FSIM :cite:`zhang:11`.

        Parameters
        ----------
        method : string, optional, (default: MSSSIM)
            The quality metric desired for this comparison.
            Options include: SSIM, MSSSIM, VIFp, FSIM
        L : scalar
            The dynamic range of the data. This value is 1 for float
            representations and 2^bitdepth for integer representations.
        """

        dictionary = {"SSIM": compute_ssim, "MSSSIM": compute_msssim,
                      "VIFp": _compute_vifp, "FSIM": _compute_fsim}
        try:
            method_func = dictionary[method]
        except KeyError:
            ValueError("That method is not implemented.")

        self.method = method

        if self.img0.ndim > 2:
            self.mets = list()
            self.maps = list()

            for i in range(self.img0.shape[2]):

                scales, mets, maps = method_func(self.img0[:, :, i],
                                                 self.img1[:, :, i],
                                                 L=L, **kwargs)

                self.scales = scales
                self.mets.append(mets)
                self.maps.append(maps)

            self.mets = np.stack(self.mets, axis=1)

            newmaps = []
            for level in range(len(self.maps[0])):
                this_level = []
                for m in self.maps:
                    this_level.append(m[level])

                this_level = np.stack(this_level, axis=2)
                newmaps.append(this_level)

            self.maps = newmaps

        else:
            self.scales, self.mets, self.maps = method_func(self.img0,
                                                            self.img1,
                                                            L=L, **kwargs)


def _join_metrics(A, B):
    """Join two image metric dictionaries."""

    for key in list(B.keys()):
        if key in A:
            A[key][0] = np.concatenate((A[key][0], B[key][0]))

            A[key][1] = np.concatenate((np.atleast_3d(A[key][1]),
                                        np.atleast_3d(B[key][1])), axis=2)

        else:
            A[key] = B[key]

    return A


def _compute_vifp(img0, img1, nlevels=5, sigma=1.2, L=None):
    """Calculate the Visual Information Fidelity (VIFp) between two images in
    in a multiscale pixel domain with scalar.

    Parameters
    ----------
    img0 : array
    img1 : array
        Two images for comparison.
    nlevels : scalar
        The number of levels to measure quality.
    sigma : scalar
        The size of the quality filter at the smallest scale.

    Returns
    -------
    metrics : dict
        A dictionary with image quality information organized by scale.
        ``metric[scale] = (mean_quality, quality_map)``
        The valid range for VIFp is (0, 1].


    .. centered:: COPYRIGHT NOTICE
    Copyright (c) 2005 The University of Texas at Austin
    All rights reserved.

    Permission is hereby granted, without written agreement and without license
    or royalty fees, to use, copy, modify, and distribute this code (the source
    files) and its documentation for any purpose, provided that the copyright
    notice in its entirety appear in all copies of this code, and the original
    source of this code, Laboratory for Image and Video Engineering (LIVE,
    http://live.ece.utexas.edu) at the University of Texas at Austin (UT
    Austin, http://www.utexas.edu), is acknowledged in any publication that
    reports research using this code. The research is to be cited in the
    bibliography as: H. R. Sheikh and A. C. Bovik, "Image Information and
    Visual Quality", IEEE Transactions on Image Processing, (to appear). IN NO
    EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR
    DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
    OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY
    OF TEXAS AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE
    UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
    AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON
    AN "AS IS" BASIS, AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION
    TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
    .. centered:: END COPYRIGHT NOTICE
    """
    _full_reference_input_check(img0, img1, sigma, nlevels, L)

    sigmaN_sq = 2  # used to tune response
    eps = 1e-10

    scales = np.zeros(nlevels)
    mets = np.zeros(nlevels)
    maps = [None] * nlevels

    for level in range(0, nlevels):
        # Downsample (using ndimage.zoom to prevent sampling bias)
        if (level > 0):
            img0 = ndimage.zoom(img0, 1/2)
            img1 = ndimage.zoom(img1, 1/2)

        mu0 = ndimage.gaussian_filter(img0, sigma)
        mu1 = ndimage.gaussian_filter(img1, sigma)

        sigma0_sq = ndimage.gaussian_filter((img0 - mu0)**2, sigma)
        sigma1_sq = ndimage.gaussian_filter((img1 - mu1)**2, sigma)
        sigma01 = ndimage.gaussian_filter((img0 - mu0) * (img1 - mu1), sigma)

        g = sigma01 / (sigma0_sq + eps)
        sigmav_sq = sigma1_sq - g * sigma01

        # Calculate VIF
        numator = np.log2(1 + g**2 * sigma0_sq / (sigmav_sq + sigmaN_sq))
        denator = np.sum(np.log2(1 + sigma0_sq / sigmaN_sq))

        vifmap = numator / denator
        vifp = np.sum(vifmap)
        # Normalize the map because we want values between 1 and 0
        vifmap *= vifmap.size

        scale = sigma * 2**level

        scales[level] = scale
        mets[level] = vifp
        maps[level] = vifmap

    return scales, mets, maps


def _compute_fsim(img0, img1, nlevels=5, nwavelets=16, L=None):
    """FSIM Index with automatic downsampling, Version 1.0

    An implementation of the algorithm for calculating the Feature SIMilarity
    (FSIM) index was ported to Python. This implementation only considers the
    luminance component of images. For multichannel images, convert to
    grayscale first. Dynamic range should be 0-255.

    Parameters
    ----------
    img0 : array
    img1 : array
        Two images for comparison.
    nlevels : scalar
        The number of levels to measure quality.
    nwavelets : scalar
        The number of wavelets to use in the phase congruency calculation.

    Returns
    -------
    metrics : dict
        A dictionary with image quality information organized by scale.
        ``metric[scale] = (mean_quality, quality_map)``
        The valid range for FSIM is (0, 1].


    .. centered:: COPYRIGHT NOTICE
    Copyright(c) 2010 Lin ZHANG, Lei Zhang, Xuanqin Mou and David Zhang
    All Rights Reserved.
    ----------------------------------------------------------------------
    Permission to use, copy, or modify this software and its documentation
    for educational and research purposes only and without fee is here
    granted, provided that this copyright notice and the original authors'
    names appear on all copies and supporting documentation. This program
    shall not be used, rewritten, or adapted as the basis of a commercial
    software or hardware product without first obtaining permission of the
    authors. The authors make no representations about the suitability of
    this software for any purpose. It is provided "as is" without express
    or implied warranty.
    ----------------------------------------------------------------------
    Lin Zhang, Lei Zhang, Xuanqin Mou, and David Zhang,"FSIM: a feature
    similarity index for image qualtiy assessment", IEEE Transactions on Image
    Processing, vol. 20, no. 8, pp. 2378-2386, 2011.
    .. centered:: END COPYRIGHT NOTICE
    """
    _full_reference_input_check(img0, img1, 1.2, nlevels, L)
    if nwavelets < 1:
        raise ValueError('There must be at least one wavelet level.')

    Y1 = img0
    Y2 = img1

    scales = np.zeros(nlevels)
    mets = np.zeros(nlevels)
    maps = [None] * nlevels

    for level in range(0, nlevels):
        # sigma = 1.2 is approximately correct because the width of the scharr
        # and min wavelet filter (phase congruency filter) is 3.
        sigma = 1.2 * 2**level

        F = 2  # Downsample (using ndimage.zoom to prevent sampling bias)
        Y1 = ndimage.zoom(Y1, 1/F)
        Y2 = ndimage.zoom(Y2, 1/F)

        # Calculate the phase congruency maps
        [PC1, Orient1, ft1, T1] = phase.phasecongmono(Y1, nscale=nwavelets)
        [PC2, Orient2, ft2, T2] = phase.phasecongmono(Y2, nscale=nwavelets)

        # Calculate the gradient magnitude map using Scharr filters
        dx = np.array([[3., 0., -3.],
                       [10., 0., -10.],
                       [3., 0., -3.]]) / 16
        dy = np.array([[3., 10., 3.],
                       [0., 0., 0.],
                       [-3., -10., -3.]]) / 16

        IxY1 = ndimage.filters.convolve(Y1, dx)
        IyY1 = ndimage.filters.convolve(Y1, dy)
        gradientMap1 = np.sqrt(IxY1**2 + IyY1**2)

        IxY2 = ndimage.filters.convolve(Y2, dx)
        IyY2 = ndimage.filters.convolve(Y2, dy)
        gradientMap2 = np.sqrt(IxY2**2 + IyY2**2)

        # Calculate the FSIM
        T1 = 0.85   # fixed and depends on dynamic range of PC values
        T2 = 160    # fixed and depends on dynamic range of GM values
        PCSimMatrix = (2 * PC1 * PC2 + T1) / (PC1**2 + PC2**2 + T1)
        gradientSimMatrix = ((2 * gradientMap1 * gradientMap2 + T2) /
                             (gradientMap1**2 + gradientMap2**2 + T2))
        PCm = np.maximum(PC1, PC2)
        FSIMmap = gradientSimMatrix * PCSimMatrix
        FSIM = np.sum(FSIMmap * PCm) / np.sum(PCm)

        scales[level] = sigma
        mets[level] = FSIM
        maps[level] = FSIMmap

    return scales, mets, maps


def compute_msssim(img0, img1, nlevels=5, sigma=1.2, L=1.0, K=(0.01, 0.03),
                   alpha=4, beta_gamma=None):
    """Multi-Scale Structural SIMilarity index (MS-SSIM).

    Parameters
    ----------
    img0 : array
    img1 : array
        Two images for comparison.
    nlevels : int
        The max number of levels to analyze
    sigma : float
        Sets the standard deviation of the gaussian filter. This setting
        determines the minimum scale at which quality is assessed.
    L : scalar
        The dynamic range of the data. This value is 1 for float
        representations and 2^bitdepth for integer representations.
    K : 2-tuple
        A list of two constants which help prevent division by zero.
    alpha : float
        The exponent which weights the contribution of the luminance term.
    beta_gamma : list
        The exponent which weights the contribution of the contrast and
        structure terms at each level.

    Returns
    -------
    metrics : dict
        A dictionary with image quality information organized by scale.
        ``metric[scale] = (mean_quality, quality_map)``
        The valid range for SSIM is [-1, 1].


    References
    ----------
    Multi-scale Structural Similarity Index (MS-SSIM)
    Z. Wang, E. P. Simoncelli and A. C. Bovik, "Multi-scale structural
    similarity for image quality assessment," Invited Paper, IEEE Asilomar
    Conference on Signals, Systems and Computers, Nov. 2003
    """
    _full_reference_input_check(img0, img1, sigma, nlevels, L)
    # The relative imporance of each level as determined by human experiment
    if beta_gamma is None:
        beta_gamma = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]) * 4
    assert nlevels < 6, "Not enough beta_gamma weights for more than 5 levels"
    scales = np.zeros(nlevels)
    maps = [None] * nlevels
    scale, luminance, ssim = compute_ssim(img0, img1, sigma=sigma, L=L,
                                          K=K, scale=sigma,
                                          alpha=alpha, beta_gamma=0)
    for level in range(0, nlevels):
        scale, mean_ssim, ssim = compute_ssim(img0, img1, sigma=sigma, L=L,
                                              K=K, scale=sigma,
                                              alpha=0,
                                              beta_gamma=beta_gamma[level])
        scales[level] = scale
        maps[level] = ndimage.zoom(ssim, 2**level, prefilter=False, order=0)
        if level == nlevels - 1:
            break
        # Downsample (using ndimage.zoom to prevent sampling bias)
        # Images become half the size
        img0 = ndimage.zoom(img0, 0.5)
        img1 = ndimage.zoom(img1, 0.5)

    map = luminance * np.nanprod(maps, axis=0)
    mean_ms_ssim = np.nanmean(map)
    return scales, mean_ms_ssim, map


def compute_ssim(img1, img2, sigma=1.2, L=1, K=(0.01, 0.03), scale=None,
                 alpha=4, beta_gamma=4):
    """Return the Structural SIMilarity index (SSIM).

    A modified version of the Structural SIMilarity index (SSIM) based on an
    implementation by Helder C. R. de Oliveira, based on the implementation by
    Antoine Vacavant, ISIT lab, antoine.vacavant@iut.u-clermont1.fr
    http://isit.u-clermont1.fr/~anvacava

    Attributes
    ----------
    img1 : array
    img2 : array
        Two images for comparison.
    sigma : float
        Sets the standard deviation of the gaussian filter. This setting
        determines the minimum scale at which quality is assessed.
    L : scalar
        The dynamic range of the data. This value is 1 for float
        representations and 2^bitdepth for integer representations.
    K : 2-tuple
        A list of two constants which help prevent division by zero.
    alpha : float
        The exponent which weights the contribution of the luminance term.
    beta_gamma : list
        The exponent which weights the contribution of the contrast and
        structure terms at each level.

    Returns
    -------
    metrics : dict
        A dictionary with image quality information organized by scale.
        ``metric[scale] = (mean_quality, quality_map)``
        The valid range for SSIM is [-1, 1].


    References
    ----------
    Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli. Image quality
    assessment: From error visibility to structural similarity. IEEE
    Transactions on Image Processing, 13(4):600--612, 2004.

    Z. Wang and A. C. Bovik. Mean squared error: Love it or leave it? - A new
    look at signal fidelity measures. IEEE Signal Processing Magazine,
    26(1):98--117, 2009.

    Silvestre-Blanes, J., & Pérez-Lloréns, R. (2011, September). SSIM and their
    dynamic range for image quality assessment. In ELMAR, 2011 Proceedings
    (pp. 93-96). IEEE.
    """
    _full_reference_input_check(img1, img2, sigma, 1, L)
    if scale is not None and scale <= 0:
        raise ValueError("Scale cannot be negative or zero.")
    assert L > 0, "L, the dynamic range must be larger than 0."
    c_1 = (K[0] * L)**2
    c_2 = (K[1] * L)**2
    # Means obtained by Gaussian filtering of inputs
    mu_1 = ndimage.filters.gaussian_filter(img1, sigma)
    mu_2 = ndimage.filters.gaussian_filter(img2, sigma)
    # Squares of means
    mu_1_sq = mu_1**2
    mu_2_sq = mu_2**2
    mu_1_mu_2 = mu_1 * mu_2
    # Variances obtained by Gaussian filtering of inputs' squares
    sigma_1_sq = ndimage.filters.gaussian_filter(img1**2, sigma) - mu_1_sq
    sigma_2_sq = ndimage.filters.gaussian_filter(img2**2, sigma) - mu_2_sq
    # Covariance
    sigma_12 = ndimage.filters.gaussian_filter(img1 * img2, sigma) - mu_1_mu_2
    # Division by zero is prevented by adding c_1 and c_2
    numerator1 = 2 * mu_1_mu_2 + c_1
    denominator1 = mu_1_sq + mu_2_sq + c_1
    numerator2 = 2 * sigma_12 + c_2
    denominator2 = sigma_1_sq + sigma_2_sq + c_2

    if (c_1 > 0) and (c_2 > 0):
        ssim_map = ((numerator1 / denominator1)**alpha *
                    (numerator2 / denominator2)**beta_gamma)
    else:
        ssim_map = np.ones(numerator1.shape)
        index = (denominator1 * denominator2 > 0)
        ssim_map[index] = ((numerator1[index]/denominator1[index])**alpha *
                           (numerator2[index]/denominator2[index])**beta_gamma)
    # Sometimes c_1 and c_2 don't do their job of stabilizing the result
    ssim_map[ssim_map > 1] = 1
    ssim_map[ssim_map < -1] = -1
    mean_ssim = np.nanmean(ssim_map)
    if scale is None:
        scale = sigma
    return scale, mean_ssim, ssim_map


def _full_reference_input_check(img0, img1, sigma, nlevels, L):
    """Checks full reference quality measures for valid inputs."""
    if nlevels <= 0:
        raise ValueError('nlevels must be >= 1.')
    if sigma < 1.2:
        raise ValueError('sigma < 1.2 is effective meaningless.')
    if np.min(img0.shape) / (2**(nlevels - 1)) < sigma * 2:
        raise ValueError("{nlevels} levels makes {shape} smaller than a filter"
                         " size of 2 * {sigma}".format(nlevels=nlevels,
                                                       shape=img0.shape,
                                                       sigma=sigma))
    if L is not None and L < 1:
        raise ValueError("Dynamic range must be >= 1.")
    if img0.shape != img1.shape:
        raise ValueError("original and reconstruction should be the " +
                         "same shape")
    if img0.ndim != 2:
        raise ValueError("This function only support 2D images.")
