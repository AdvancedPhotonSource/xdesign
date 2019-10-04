#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines standards based image quality metrics.

These methods require the reconstructed image to be of a specifically shaped
standard object such as a Siemens star or a zone plate.

.. moduleauthor:: Daniel J Ching <carterbox@users.noreply.github.com>
"""

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'compute_mtf_ffst',
    'compute_mtf_lwkj',
    'compute_nps_ffst',
    'compute_neq_d',
]

import warnings

import numpy as np
from scipy import optimize

from xdesign.geometry import Circle, Point, Line
from xdesign.phantom import HyperbolicConcentric, UnitCircle


def compute_mtf_ffst(phantom, image, Ntheta=4):
    '''Calculate the MTF using the method described in :cite:`Friedman:13`.

    .. seealso::

        :meth:`compute_mtf_lwkj`

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
    Th_bins = np.arange(
        -Th_bin_width / 2, 2 * np.pi - Th_bin_width / 2, Th_bin_width
    )
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

    LSF = -np.diff(ESF, axis=1)

    # trim the LSF so that the edge is in the center of the data
    edge_center = int(phantom.geometry.radius / R_bin_width)
    # print(edge_center)
    pad = int(LSF.shape[1] / 5)
    LSF = LSF[:, edge_center - pad:edge_center + pad + 1]
    # print(LSF)
    LSF_weighted = LSF * np.hanning(LSF.shape[1])

    # Calculate the MTF
    T = np.fft.fftshift(np.fft.fft(LSF_weighted))
    faxis = (np.arange(0, LSF.shape[1]) / LSF.shape[1] - 0.5) / R_bin_width
    nyquist = 0.5 * image.shape[0]

    MTF = np.abs(T)
    bin_centers = Th_bins + Th_bin_width / 2

    return faxis, MTF, bin_centers


def compute_mtf_lwkj(image, n_sectors, n_radii=100):
    """Calculate the MTF using the modulated Siemens Star method in
    :cite:`loebich2007digital`.

    .. seealso::

        :meth:`compute_mtf_ffst`

    Parameters
    ----------
    image : ndarray
        A centered image of a Siemens star.
    n_sectors: int >= 2
        The number of spokes/blades on the star. i.e. the number of
        light/dark pairs.

    Returns
    -------
    frequency : array
        The spatial frequency in cycles per pixel length.
    mtf : array
        The MTF values for each frequency.

    .. seealso::
        :meth:`compute_mtf_ffst`
    """
    assert image.shape[0] == image.shape[1], "image should be square."
    # Determine which radii to sample. Frequencies are limited by
    # the radius of the image and the Nyqust fequency (1/2).
    frequency = np.linspace(
        0.5,
        n_sectors / (np.pi * image.shape[0]),
        n_radii,
        endpoint=False,
    )
    # Convert frequency into fractional radii; assume square image
    fradii = n_sectors / (np.pi * frequency * image.shape[0])
    line, theta = get_line_at_radius(image, fradii)
    mtf = fit_sinusoid(line, theta, n_sectors)
    return frequency, mtf


def get_line_at_radius(image, fradius, N=None):
    """Return an Nx1 array of the values of the image at a radius.

    Parameters
    ----------
    image : :py:class:`numpy.ndarray`
        A centered image of the Siemens star.
    fradius : (M, ) :py:class:`numpy.array_like`
        The fractional radii of the image at which to extract lines.
        Given as a floats in the range (0, 1).
    N : int >= PI * image_width
        The number of points to sample along each line.

    Returns
    -------
    line : (N, M) :py:class:`numpy.ndarray`
        The values from image at each radius.
    theta : (N, 1) :py:class:`numpy.ndarray`
        The angles that were sampled [radians].

    Raises
    ------
    ValueError
        If any value of `fradius` is not between 0 and 1.
    """
    fradius = np.asanyarray(fradius)
    if np.any(fradius <= 0) or np.any(1 <= fradius):
        raise ValueError('fradius must be between 0 and 1.')
    # set the number of sample to pi * d in order to get good sampling
    image_width = np.min(image.shape)
    if N is None:
        N = int(np.pi * image_width)
    else:
        N = max(N, int(np.pi * image_width))
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
    image_half = image_width / 2
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

    Parameters
    ----------
    value : NxM ndarray
        The value of the function at N angles and M radii
    angle : Nx1 ndarray
        The N angles at which the function was sampled
    f : scalar
        The expected angular frequency; the number of black/white pairs in
        the Siemens star.
    p0 : list, optional
        The initial guesses for the parameters.

    Returns
    -------
    MTFR: 1xM ndarray
        The modulation part of the MTF at each of the M radii
    """
    M = value.shape[1]

    # Distance to the target function
    def errorfunc(p, x, y):
        return periodic_function(p, x) - y

    time = np.linspace(0, 2 * np.pi, 100)

    MTFR = np.ndarray((1, M))
    x = (f * angle).squeeze()
    for radius in range(0, M):
        p1, success = optimize.leastsq(
            errorfunc, p0[:], args=(x, value[:, radius])
        )

        MTFR[:, radius] = np.sqrt(p1[1]**2 + p1[2]**2) / p1[0]

    # cap the MTF at unity
    MTFR[MTFR > 1.] = 1.
    assert (not np.any(MTFR < 0)), MTFR
    assert (MTFR.shape == (1, M)), MTFR.shape
    return MTFR


def periodic_function(p, x):
    """A periodic function for fitting to the spokes of the Siemens Star.

    Parameters
    ----------
    p[0] : scalar
        the mean of the function
    p[1], p[2] : scalar
        the amplitudes of the function
    x : Nx1 ndarray
        the angular frequency multiplied by the angles for the function.
        w * theta
    w : scalar
        the angular frequency; the number of black/white pairs in the Siemens
        star. i.e. half the number of spokes
    theta : Nx1 ndarray
        input angles for the function

    Returns
    -------
    value : Nx1 array
        the values of the function at phi; cannot return NaNs.
    """
    # x = w * theta
    value = p[0] + p[1] * np.sin(x) + p[2] * np.cos(x)
    assert (value.shape == x.shape), (value.shape, x.shape)
    assert (not np.any(np.isnan(value)))
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

    Returns
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

    resolution = image.shape[0]  # [pixels/length]
    # cut out uniform region (square circumscribed by unit circle)
    i_half = int(image.shape[0] / 2)  # half image
    # half of the square inside the circle
    s_half = int(image.shape[0] * phantom.geometry.radius / np.sqrt(2))
    unif_region = image[i_half - s_half:i_half + s_half, i_half -
                        s_half:i_half + s_half]

    # zero-mean normalization
    unif_region = unif_region - np.mean(unif_region)

    # 2D fourier-transform
    unif_region = np.fft.fftshift(np.fft.fft2(unif_region))
    # squared modulus / squared complex norm
    NPS = np.abs((unif_region))**2  # [attenuation^2]

    # Calculate axis labels
    # TODO@dgursoy is this frequency scaling correct?
    x = y = (np.arange(0, unif_region.shape[0]) / unif_region.shape[0] -
             0.5) * image.shape[0]
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
            bucket = np.logical_and(bucket, mu_a < mu_b[i + 1])

        if NPS[bucket].size > 0:
            NPS_binned[0, i] = np.sum(NPS[bucket])

    NEQ = MTF / np.sqrt(NPS_binned)  # or something similiar

    return mu_b, NEQ
