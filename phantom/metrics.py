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
import scipy.ndimage
import logging

logger = logging.getLogger(__name__)

__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['background_mask']

def background_mask(phantom, shape):
    """Returns the background mask.

    Parameters
    ----------
    phantom : Phantom
    shape : ndarray

    Returns
    -------
    ndarray, bool
        True if pixel belongs to (an assumed) background.
    """
    dx, dy = shape
    _x = np.arange(0, 1, 1 / dx)
    _y = np.arange(0, 1, 1 / dy)
    px, py = np.meshgrid(_x, _y)

    mask = np.zeros(shape, dtype=np.bool)
    mask += (px - 0.5)**2 + (py - 0.5)**2 < 0.5**2
    for m in range(phantom.population):
        x = phantom.feature[m].center.x
        y = phantom.feature[m].center.y
        rad = phantom.feature[m].radius
        mask -= (px - x)**2 + (py - y)**2 < rad**2
    return mask

def _compute_ssim(im1, im2, l=255, filtersize=(11,11), sigma=None):
    """
    This is a modified version of SSIM based on implementation by
    Helder C. R. de Oliveira, based on the version of:
    Antoine Vacavant, ISIT lab, antoine.vacavant@iut.u-clermont1.fr,
    http://isit.u-clermont1.fr/~anvacava

    References:
        [1] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli.
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13(4):600--612, 2004.

        [2] Z. Wang and A. C. Bovik.
        Mean squared error: Love it or leave it? - A new look at signal fidelity measures.
        IEEE Signal Processing Magazine, 26(1):98--117, 2009.

    Attributes
    ----------
    im1 : scalar
    im2 : scalar
    l : scalar, default = 255
        The dynamic range of the images. i.e 2^bitdepth-1
    filtersize : list, optional
        The dimensions of the filter to use.
    sigma : list, optional
        The standard deviation of the gaussian filter. If not specified,
        means are calculated using uniform square filters.

    Returns
    ----------
    index : scalar
        The mean structural similarity
    ssim_map : Array
        The local measures of structural Similiarity
    """

    # k1,k2 & c1,c2 depend on L (width of color map)
    k_1 = 0.01
    c_1 = (k_1*l)**2
    k_2 = 0.03
    c_2 = (k_2*l)**2

    window = np.ones(filtersize)
    if sigma is not None:
        window = _gauss_2d(filtersize, sigma)
    # Normalization
    window /= np.sum(window)

    # Convert image matrices to double precision (like in the Matlab version)
    im1 = im1.astype(np.float)
    im2 = im2.astype(np.float)

    # TODO: Replace convolve with uniform and gaussian filtering methods because they can probably be optimized to be faster.
    # Means obtained by Gaussian filtering of inputs
    mu_1 = scipy.ndimage.filters.convolve(im1, window)
    mu_2 = scipy.ndimage.filters.convolve(im2, window)

    # Squares of means
    mu_1_sq = mu_1**2
    mu_2_sq = mu_2**2
    mu_1_mu_2 = mu_1 * mu_2

    # Squares of input matrices
    im1_sq = im1**2
    im2_sq = im2**2
    im12 = im1*im2

    # Variances obtained by Gaussian filtering of inputs' squares
    sigma_1_sq = scipy.ndimage.filters.convolve(im1_sq, window)
    sigma_2_sq = scipy.ndimage.filters.convolve(im2_sq, window)

    # Covariance
    sigma_12 = scipy.ndimage.filters.convolve(im12, window)

    # Centered squares of variances
    sigma_1_sq -= mu_1_sq
    sigma_2_sq -= mu_2_sq
    sigma_12 -= mu_1_mu_2

    if (c_1 > 0) & (c_2 > 0):
        ssim_map = ((2*mu_1_mu_2 + c_1) * (2*sigma_12 + c_2)) / ((mu_1_sq + mu_2_sq + c_1) * (sigma_1_sq + sigma_2_sq + c_2))
    else:
        numerator1 = 2 * mu_1_mu_2 + c_1
        numerator2 = 2 * sigma_12 + c_2

        denominator1 = mu_1_sq + mu_2_sq + c_1
        denominator2 = sigma_1_sq + sigma_2_sq + c_2

        ssim_map = np.ones(mu_1.size)

        index = (denominator1 * denominator2 > 0)

        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    # return MSSIM
    index = np.mean(ssim_map)

    return (index, ssim_map)

def _gauss_2d(shape=(3, 3), sigma=0.5):
    """
    Code from Stack Overflow's thread
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
