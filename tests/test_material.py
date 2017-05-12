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

import numpy as np
from numpy.testing import assert_allclose, assert_raises, assert_equal
import matplotlib.pyplot as plt
import warnings
import os.path

from xdesign.phantom import *
from xdesign.material import *
from xdesign.plot import *

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def _plot_both(ref, target):
    """Plot two images to compare them."""
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(ref, cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(target, cmap='viridis')
    plt.colorbar()
    plt.show(block=False)


def _save_and_load(phantom_class, args=[]):
    """Test whether the saved and loaded phantoms match."""
    saved_phantom = 'tests/{}{}.txt'.format(phantom_class.__name__, args)

    np.random.seed(0)
    p0 = phantom_class(*args)

    if not os.path.isfile(saved_phantom):
        save_phantom(p0, saved_phantom)

    p1 = load_phantom(saved_phantom)

    refere = discrete_phantom(p0, 200, uniform=False)
    target = discrete_phantom(p1, 200, uniform=False)

    _plot_both(refere, target)

    assert_equal(target, refere,
                 "{}({}) changes on load.".format(phantom_class.__name__,
                                                  args))


def test_HyperbolicCocentric():
    _save_and_load(HyperbolicConcentric)


def test_DynamicRange():
    warnings.filterwarnings("ignore", "The Square*", UserWarning)
    _save_and_load(DynamicRange, [10, True])
    _save_and_load(DynamicRange, [10, False])


def test_Soil():
    warnings.filterwarnings("ignore", "Reached*", RuntimeWarning)
    _save_and_load(Soil)


# def test_Foam():
#     warnings.filterwarnings("ignore", "Reached*", RuntimeWarning)
#     _save_and_load(Foam)


def test_XDesignDefault():
    _save_and_load(XDesignDefault)
    p = XDesignDefault()
    sidebyside(p)


if __name__ == '__main__':
    test_XDesignDefault()
    plt.show(block=True)
