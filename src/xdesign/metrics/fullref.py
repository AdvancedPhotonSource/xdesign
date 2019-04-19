#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines full-referene image quality metricsself.

These methods require a ground truth in order to make a quality assessment.

.. moduleauthor:: Daniel J Ching <carterbox@users.noreply.github.com>
"""

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'pcc',
    'ImageQuality',
    'ssim',
    'msssim',
]

import warnings

import numpy as np
from scipy import ndimage

warnings.filterwarnings(
    'ignore',
    'From scipy 0\.13\.0, the output shape of zoom\(\) '
    'is calculated with round\(\) instead of int\(\)'
)


def pcc(A, B, masks=None):
    """Return the Pearson product-moment correlation coefficients (PCC).

    Parameters
    -------------
    A, B : ndarray
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

    def quality(self, method="MSSSIM", L=1.0, **kwargs):
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
        dictionary = {
            "SSIM": ssim,
            "MSSSIM": msssim,
            "VIFp": vifp,
            "FSIM": fsim
        }
        try:
            method_func = dictionary[method]
        except KeyError:
            ValueError("That method is not implemented.")

        self.method = method

        if self.img0.ndim > 2:
            self.mets = list()
            self.maps = list()

            for i in range(self.img0.shape[2]):

                scales, mets, maps = method_func(
                    self.img0[:, :, i], self.img1[:, :, i], L=L, **kwargs
                )

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
            self.scales, self.mets, self.maps = method_func(
                self.img0, self.img1, L=L, **kwargs
            )


def _join_metrics(A, B):
    """Join two image metric dictionaries."""
    for key in list(B.keys()):
        if key in A:
            A[key][0] = np.concatenate((A[key][0], B[key][0]))

            A[key][1] = np.concatenate(
                (np.atleast_3d(A[key][1]), np.atleast_3d(B[key][1])), axis=2
            )

        else:
            A[key] = B[key]

    return A


def vifp(img0, img1, nlevels=5, sigma=1.2, L=None):
    """Return the Visual Information Fidelity (VIFp) of two images.

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

    Copyright (c) 2005 The University of Texas at Austin. All rights reserved.

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
            img0 = ndimage.zoom(img0, 0.5)
            img1 = ndimage.zoom(img1, 0.5)

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


# def fsim(img0, img1, nlevels=5, nwavelets=16, L=None):
#     """FSIM Index with automatic downsampling, Version 1.0
#
#     An implementation of the algorithm for calculating the Feature SIMilarity
#     (FSIM) index was ported to Python. This implementation only considers the
#     luminance component of images. For multichannel images, convert to
#     grayscale first. Dynamic range should be 0-255.
#
#     Parameters
#     ----------
#     img0 : array
#     img1 : array
#         Two images for comparison.
#     nlevels : scalar
#         The number of levels to measure quality.
#     nwavelets : scalar
#         The number of wavelets to use in the phase congruency calculation.
#
#     Returns
#     -------
#     metrics : dict
#         A dictionary with image quality information organized by scale.
#         ``metric[scale] = (mean_quality, quality_map)``
#         The valid range for FSIM is (0, 1].
#
#
#     References
#     ----------
#     Lin Zhang, Lei Zhang, Xuanqin Mou, and David Zhang,"FSIM: a feature
#     similarity index for image qualtiy assessment", IEEE Transactions on Image
#     Processing, vol. 20, no. 8, pp. 2378-2386, 2011.
#
#     .. centered:: COPYRIGHT NOTICE
#
#     Copyright (c) 2010 Lin ZHANG, Lei Zhang, Xuanqin Mou and David Zhang.
#     All Rights Reserved.
#
#     Permission to use, copy, or modify this software and its documentation
#     for educational and research purposes only and without fee is here
#     granted, provided that this copyright notice and the original authors'
#     names appear on all copies and supporting documentation. This program
#     shall not be used, rewritten, or adapted as the basis of a commercial
#     software or hardware product without first obtaining permission of the
#     authors. The authors make no representations about the suitability of
#     this software for any purpose. It is provided "as is" without express
#     or implied warranty.
#
#     .. centered:: END COPYRIGHT NOTICE
#     """
#     _full_reference_input_check(img0, img1, 1.2, nlevels, L)
#     if nwavelets < 1:
#         raise ValueError('There must be at least one wavelet level.')
#
#     Y1 = img0
#     Y2 = img1
#
#     scales = np.zeros(nlevels)
#     mets = np.zeros(nlevels)
#     maps = [None] * nlevels
#
#     for level in range(0, nlevels):
#         # sigma = 1.2 is approximately correct because the width of the scharr
#         # and min wavelet filter (phase congruency filter) is 3.
#         sigma = 1.2 * 2**level
#
#         F = 2  # Downsample (using ndimage.zoom to prevent sampling bias)
#         Y1 = ndimage.zoom(Y1, 1/F)
#         Y2 = ndimage.zoom(Y2, 1/F)
#
#         # Calculate the phase congruency maps
#         [PC1, Orient1, ft1, T1] = phase.phasecongmono(Y1, nscale=nwavelets)
#         [PC2, Orient2, ft2, T2] = phase.phasecongmono(Y2, nscale=nwavelets)
#
#         # Calculate the gradient magnitude map using Scharr filters
#         dx = np.array([[3., 0., -3.],
#                        [10., 0., -10.],
#                        [3., 0., -3.]]) / 16
#         dy = np.array([[3., 10., 3.],
#                        [0., 0., 0.],
#                        [-3., -10., -3.]]) / 16
#
#         IxY1 = ndimage.filters.convolve(Y1, dx)
#         IyY1 = ndimage.filters.convolve(Y1, dy)
#         gradientMap1 = np.sqrt(IxY1**2 + IyY1**2)
#
#         IxY2 = ndimage.filters.convolve(Y2, dx)
#         IyY2 = ndimage.filters.convolve(Y2, dy)
#         gradientMap2 = np.sqrt(IxY2**2 + IyY2**2)
#
#         # Calculate the FSIM
#         T1 = 0.85   # fixed and depends on dynamic range of PC values
#         T2 = 160    # fixed and depends on dynamic range of GM values
#         PCSimMatrix = (2 * PC1 * PC2 + T1) / (PC1**2 + PC2**2 + T1)
#         gradientSimMatrix = ((2 * gradientMap1 * gradientMap2 + T2) /
#                              (gradientMap1**2 + gradientMap2**2 + T2))
#         PCm = np.maximum(PC1, PC2)
#         FSIMmap = gradientSimMatrix * PCSimMatrix
#         FSIM = np.sum(FSIMmap * PCm) / np.sum(PCm)
#
#         scales[level] = sigma
#         mets[level] = FSIM
#         maps[level] = FSIMmap
#
#     return scales, mets, maps


def msssim(
    img0,
    img1,
    nlevels=5,
    sigma=1.2,
    L=1.0,
    K=(0.01, 0.03),
    alpha=4,
    beta_gamma=None
):
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
    scale, luminance, ssim_map = ssim(
        img0,
        img1,
        sigma=sigma,
        L=L,
        K=K,
        scale=sigma,
        alpha=alpha,
        beta_gamma=0
    )
    original_shape = np.array(img0.shape)
    for level in range(0, nlevels):
        scale, ssim_mean, ssim_map = ssim(
            img0,
            img1,
            sigma=sigma,
            L=L,
            K=K,
            scale=sigma,
            alpha=0,
            beta_gamma=beta_gamma[level]
        )
        # Always take the direct ratio between original and downsampled maps
        # to prevent resizing mismatch for odd sizes
        ratio = original_shape / np.array(ssim_map.shape)
        scales[level] = scale * ratio[0]
        maps[level] = ndimage.zoom(ssim_map, ratio, prefilter=False, order=0)

        if level == nlevels - 1:
            break
        # Downsample (using ndimage.zoom to prevent sampling bias)
        # Images become half the size
        img0 = ndimage.zoom(img0, 0.5)
        img1 = ndimage.zoom(img1, 0.5)

    map = luminance * np.nanprod(maps, axis=0)
    ms_ssim_mean = np.nanmean(map)
    return scales, ms_ssim_mean, map


def ssim(
    img1,
    img2,
    sigma=1.2,
    L=1,
    K=(0.01, 0.03),
    scale=None,
    alpha=4,
    beta_gamma=4
):
    """Return the Structural SIMilarity index (SSIM) of two images.

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
        The dynamic range of the data. The difference between the
        minimum and maximum of the data: 2^bitdepth for integer
        representations.
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
        with np.errstate(invalid='ignore'):
            ssim_map = ((numerator1 / denominator1)**alpha *
                        (numerator2 / denominator2)**beta_gamma)
    else:
        ssim_map = np.ones(numerator1.shape)
        index = (denominator1 * denominator2 > 0)
        ssim_map[index] = ((numerator1[index] / denominator1[index])**alpha *
                           (numerator2[index] / denominator2[index])**
                           beta_gamma)
    # Sometimes c_1 and c_2 don't do their job of stabilizing the result
    with np.errstate(invalid='ignore'):
        ssim_map[ssim_map > 1] = 1
        ssim_map[ssim_map < -1] = -1
    ssim_mean = np.nanmean(ssim_map)
    if scale is None:
        scale = sigma
    return scale, ssim_mean, ssim_map


def _full_reference_input_check(img0, img1, sigma, nlevels, L):
    """Checks full reference quality measures for valid inputs."""
    if nlevels <= 0:
        raise ValueError('nlevels must be >= 1.')
    if sigma < 1.2:
        raise ValueError('sigma < 1.2 is effective meaningless.')
    if np.min(img0.shape) / (2**(nlevels - 1)) < sigma * 2:
        raise ValueError(
            "{nlevels} levels makes {shape} smaller than a filter"
            " size of 2 * {sigma}".format(
                nlevels=nlevels, shape=img0.shape, sigma=sigma
            )
        )
    if L is not None and L < 1:
        raise ValueError("Dynamic range must be >= 1.")
    if img0.shape != img1.shape:
        raise ValueError(
            "original and reconstruction should be the " + "same shape"
        )
