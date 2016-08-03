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
import warnings

logger = logging.getLogger(__name__)

__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['ImageQuality','background_mask','compute_quality']

class ImageQuality(object):
    """Stores information about image quality"""
    def __init__(self, original, reconstruction):
        self.orig = original.astype(np.float)
        self.recon = reconstruction.astype(np.float)
        self.qualities = []
        self.maps = []
        self.scales = []

    def __str__(self):
        return "QUALITY: " + str(self.qualities) + "\nSCALES: " + str(self.scales)

    def __add__(self, other):
        self.qualities += other.qualities
        self.maps += other.maps
        self.scales += other.scales
        return self

    def add_quality(self,quality,scale,maps=None):
        '''
        Attributes
        -----------
        quality : scalar, list
            The average quality for the image
        map : array, list of arrays, optional
            the local quality rating across the image
        scale : scalar, list
            the size scale at which the quality was calculated
        '''
        if type(quality) is list:
            self.qualities += quality
            self.scales += scale
            if maps is None:
                maps = [None]*len(quality)
            self.maps += maps
        else:
            self.qualities.append(quality)
            self.scales.append(scale)
            self.maps.append(maps)

    def sort(self):
        """Sorts the qualities by scale. #STUB"""
        warnings.warn("ImageQuality.sort is not yet implmemented.")

def compute_quality(reference,reconstructions,method="MSSSIM", L=1):
    """
    Computes image quality metrics for each of the reconstructions.

    Parameters
    ---------
    reference : array
        the discrete reference image. In a future release, we will
        determine the best way to compare a continuous domain to a discrete
        reconstruction.
    reconstructions : list of arrays
        A list of discrete reconstructions
    method : string, optional
        The quality metric desired for this comparison.
        Options include: SSIM, MSSSIM
    L : scalar
        The dynamic range of the data. This value is 1 for float representations
        and 2^bitdepth for integer representations.

    Returns
    ---------
    metrics : list of ImageQuality
    """
    if not (type(reconstructions) is list):
        reconstructions = [reconstructions]

    dictionary = {"SSIM": _compute_ssim, "MSSSIM": _compute_msssim, "VIFp": _compute_vifp}
    method = dictionary[method]

    metrics = []
    for image in reconstructions:
        IQ = ImageQuality(reference, image)
        IQ = method(IQ, L=L)
        metrics.append(IQ)

    return metrics

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

def _compute_vifp(imQual, nlevels=5, sigma=1.2, L=None):
    """ Calculates the Visual Information Fidelity (VIFp) between two images in
    in a multiscale pixel domain with scalar.

    -----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
    Copyright (c) 2005 The University of Texas at Austin
    All rights reserved.

    Permission is hereby granted, without written agreement and without license or
    royalty fees, to use, copy, modify, and distribute this code (the source files)
    and its documentation for any purpose, provided that the copyright notice in
    its entirety appear in all copies of this code, and the original source of this
    code, Laboratory for Image and Video Engineering
    (LIVE, http://live.ece.utexas.edu) at the University of Texas at Austin
    (UT Austin, http://www.utexas.edu), is acknowledged in any publication that
    reports research using this code. The research is to be cited in the
    bibliography as:

    H. R. Sheikh and A. C. Bovik, "Image Information and Visual Quality", IEEE
    Transactions on Image Processing, (to appear).

    IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR
    DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
    THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS
    AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
    FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS
    IS" BASIS, AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE
    MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
    -----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------

    Parameters
    -----------

    Returns
    -----------

    """
    ref = imQual.orig
    dist = imQual.recon

    sigmaN_sq = 2 # used to tune response
    eps = 1e-10

    for level in range(0, nlevels):
        # Downsampling
        if (level > 0):
            ref = scipy.ndimage.uniform_filter(ref, 2)
            dist = scipy.ndimage.uniform_filter(dist, 2)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        # TODO: @Daniel convolving with a low pass 11x11 normalized gaussian
        mu1 = scipy.ndimage.gaussian_filter(ref, sigma)
        mu2 = scipy.ndimage.gaussian_filter(dist, sigma)

        sigma1_sq = scipy.ndimage.gaussian_filter((ref-mu1)**2,         sigma)
        sigma2_sq = scipy.ndimage.gaussian_filter((dist-mu2)**2,        sigma)
        sigma12   = scipy.ndimage.gaussian_filter((ref-mu1)*(dist-mu2), sigma)

        g = sigma12 / (sigma1_sq + eps)
        sigmav_sq = sigma2_sq - g * sigma12

        # Calculate VIF
        numator = np.log2(1 + g**2 * sigma1_sq / (sigmav_sq + sigmaN_sq))
        denator = np.sum(np.log2(1 + sigma1_sq / sigmaN_sq))

        vifmap = numator/denator
        vifp = np.sum(vifmap)
        # Normalize the map because we want values between 1 and 0
        vifmap *= vifmap.size

        scale = sigma*2**level
        imQual.add_quality(vifp,scale,maps=vifmap)

    return imQual

def _compute_msssim(imQual, nlevels=5, sigma=1.2, L=1, K=(0.01,0.03)):
    '''
    Multi-scale Structural Similarity Index (MS-SSIM)
    Z. Wang, E. P. Simoncelli and A. C. Bovik, "Multi-scale structural similarity
    for image quality assessment," Invited Paper, IEEE Asilomar Conference on
    Signals, Systems and Computers, Nov. 2003

    Paper can be found at: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

    Parameters
    -------------
    imQual : ImageQuality
    nlevels : int
        The max number of levels to analyze
    sigma : float
        Sets the standard deviation of the gaussian filter. This setting
        determines the minimum scale at which quality is assessed.
    L : scalar
        The dynamic range of the data. This value is 1 for float representations
        and 2^bitdepth for integer representations.
    K : 2-tuple
        A list of two constants which help prevent division by zero.

    Returns
    --------------
    imQual : ImageQuality

    '''
    img1 = imQual.orig
    img2 = imQual.recon

    # CHECK INPUTS FOR VALIDITY
    # assert that there is at least one level requested
    assert(nlevels > 0)
    # assert that the image never becomes smaller than the filter
    (M,N) = img1.shape
    min_img_width = min(M,N)/(2**(nlevels-1))
    max_filter_width = sigma*2
    assert(min_img_width >= max_filter_width)

    # The relative imporance of each level as determined by human experimentation
    #weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    for l in range(0,nlevels):
        imQual += _compute_ssim(ImageQuality(img1,img2), sigma=sigma, L=L, K=K, scale=sigma*2**l);
        if l == nlevels-1: break

        # Apply lowpass filter retain size, reflect at edges
        filtered_im1 = scipy.ndimage.filters.uniform_filter(img1, size=2)
        filtered_im2 = scipy.ndimage.filters.uniform_filter(img2, size=2)
        # Downsample by factor of two using numpy slicing
        img1 = filtered_im1[::2,::2];
        img2 = filtered_im2[::2,::2];

    return imQual

def _compute_ssim(imQual, sigma=1.2, L=1, K=(0.01,0.03), scale=None):
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
    imQual : ImageQuality
    L : scalar
        The dynamic range of the data. This value is 1 for float representations
        and 2^bitdepth for integer representations.
    sigma : list, optional
        The standard deviation of the gaussian filter.
    Returns
    ----------
    imQual : ImageQuality
    """
    if scale == None:
        scale = sigma

    c_1 = (K[0]*L)**2
    c_2 = (K[1]*L)**2

    # Convert image matrices to double precision (like in the Matlab version)
    img1 = imQual.orig
    img2 = imQual.recon

    # Means obtained by Gaussian filtering of inputs
    mu_1 = scipy.ndimage.filters.gaussian_filter(img1, sigma)
    mu_2 = scipy.ndimage.filters.gaussian_filter(img2, sigma)

    # Squares of means
    mu_1_sq = mu_1**2
    mu_2_sq = mu_2**2
    mu_1_mu_2 = mu_1 * mu_2

    # Squares of input matrices
    im1_sq = img1**2
    im2_sq = img2**2
    im12 = img1*img2

    # Variances obtained by Gaussian filtering of inputs' squares
    sigma_1_sq = scipy.ndimage.filters.gaussian_filter(im1_sq, sigma)
    sigma_2_sq = scipy.ndimage.filters.gaussian_filter(im2_sq, sigma)

    # Covariance
    sigma_12 = scipy.ndimage.filters.gaussian_filter(im12, sigma)

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
        ssim_map[index] = (numerator1[index] / denominator1[index])**4

    # return SSIM
    index = np.mean(ssim_map)
    imQual.add_quality(index, scale, maps=ssim_map)
    return imQual
