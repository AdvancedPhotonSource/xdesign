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

from xdesign.phantom import *
from xdesign.material import *
from xdesign.plot import *
from numpy.testing import assert_allclose, assert_raises, assert_equal
import numpy as np
import scipy
import matplotlib.pyplot as plt
import warnings

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def _plot_both(ref, target):
    plt.figure()
    plt.imshow(ref, cmap='viridis')
    plt.colorbar()
    plt.figure()
    plt.imshow(target, cmap='viridis')
    plt.colorbar()
    plt.show(block=True)


def test_HyperbolicCocentric():
    p0 = Phantom()
    p0.load('tests/HyperbolicConcentric.txt')
    ref = discrete_phantom(p0, 200, uniform=False)

    np.random.seed(0)
    p = HyperbolicConcentric()
    target = discrete_phantom(p, 200, uniform=False)

    # _plot_both(ref, target)
    assert_equal(target, ref,
                 "Default HyperbolicConcentric phantom has changed.")


def test_DynamicRange():
    for i in range(0, 2):
        p0 = Phantom()
        p0.load('tests/DynamicRange'+str(i)+'.txt')
        ref = discrete_phantom(p0, 100)

        np.random.seed(0)
        p = DynamicRange(jitter=i)
        target = discrete_phantom(p, 100)
        # _plot_both(ref, target)
        assert_equal(target, ref, "Default DynamicRange" + str(i) +
                                  " phantom has changed.")


def test_Soil():
    warnings.filterwarnings("ignore", "Reached*", RuntimeWarning)
    p0 = Phantom()
    p0.load('tests/Soil.txt')
    ref = discrete_phantom(p0, 100)

    np.random.seed(0)
    p = Soil()
    target = discrete_phantom(p, 100)
    # _plot_both(ref, target)
    assert_equal(target, ref, "Default Soil phantom has changed.")


def test_Foam():
    warnings.filterwarnings("ignore", "Reached*", RuntimeWarning)
    p0 = Phantom()
    p0.load('tests/Foam.txt')
    ref = discrete_phantom(p0, 100)

    np.random.seed(0)
    p = Foam()
    target = discrete_phantom(p, 100)
    # _plot_both(ref, target)
    assert_equal(target, ref, "Default Foam phantom has changed.")


def test_XDesignDefault():
    p = XDesignDefault()
    sidebyside(p)
    plt.show(block=True)


if __name__ == '__main__':
    test_XDesignDefault()
