#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2016, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
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

from numpy.testing import assert_equal, assert_almost_equal
import numpy as np
import scipy

from xdesign.plot import plot_metrics
from xdesign.metrics import (_compute_ssim, _compute_vifp, _compute_fsim,
                             ImageQuality)


__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

img1 = scipy.ndimage.imread("tests/cameraman.png")
img4 = scipy.ndimage.imread("tests/cameraman_mixed1.png")


def test_SSIM_same_image_is_unity():
    scales, mets, maps = _compute_ssim(img1, img1)
    assert_equal(mets, 1, err_msg="Mean is not unity.")
    assert_equal(maps, np.ones(img1.shape),
                 err_msg="local metrics are not unity.")
    assert_equal(img1.shape, maps.shape,
                 err_msg="SSIMs map not the same size as input")


def test_VIFp_same_image_is_unity():
    scales, mets, maps = _compute_vifp(img1, img1)
    assert_almost_equal(mets, 1, err_msg="Mean is not unity.")
    # assert_equal(IQ.maps,1,err_msg="local metrics are not unity.")


def test_FSIM_same_image_is_unity():
    scales, mets, maps = _compute_fsim(img1, img1)
    assert_almost_equal(mets, 1., err_msg="Mean is not unity.")
    # assert_almost_equal(IQ.maps, np.ones(len(IQ.maps)),
    #                     err_msg="local metrics are not unity.")


def test_compute_quality_cameraman():
    IQ = ImageQuality(img1, img4)
    IQ.compute_quality(method="VIFp", L=256)
    plot_metrics(IQ)
    IQ.compute_quality(method="FSIM", L=256)
    plot_metrics(IQ)
    IQ.compute_quality(method="MSSSIM", L=256)
    plot_metrics(IQ)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_compute_quality_cameraman()
    plt.show(block=True)
