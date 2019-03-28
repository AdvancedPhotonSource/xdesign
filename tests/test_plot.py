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

import matplotlib.pyplot as plt
import numpy as np
import scipy
from xdesign import *
from numpy.testing import *
import warnings
from copy import deepcopy

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


p = XDesignDefault()


def test_plot_phantom_plain():
    plot_phantom(p)
    plt.suptitle('test_plot_phantom_plain')


def test_plot_phantom_color_map():
    plot_phantom(p, labels=True, c_props=['linear_attenuation'])
    plt.suptitle('test_plot_phantom_color_map')


def test_sidebyside():
    sidebyside(p, size=100)
    plt.suptitle('test_sidebyside')


def test_discrete_geometry():

    plt.figure()
    plt.suptitle('test_discrete_geometry')

    # define the points of the mesh
    a = Point([0.6, 0.6])
    b = Point([0.6, 0.4])
    c = Point([0.8, 0.4])
    d = (a + c) / 2
    e = (a + b) / 2

    t0 = Triangle(deepcopy(b), deepcopy(c), deepcopy(d))

    corner, patch = discrete_geometry(t0, 1/100)
    plt.subplot(131)
    plt.imshow(patch)
    plt.title("single triangle")

    # construct and reposition the mesh
    m0 = Mesh()
    m0.append(Triangle(deepcopy(a), deepcopy(e), deepcopy(d)))
    m0.append(Triangle(deepcopy(b), deepcopy(d), deepcopy(e)))

    corner, patch = discrete_geometry(m0, 1/100)
    plt.subplot(132)
    plt.imshow(patch)
    plt.title("double triangle mesh")

    # define the circles
    m1 = Mesh()
    m1.append(Circle(Point([0.3, 0.5]), radius=0.1))
    m1.append(-Circle(Point([0.3, 0.5]), radius=0.02))

    corner, patch = discrete_geometry(m1, 1/100)
    plt.subplot(133)
    plt.imshow(patch)
    plt.title("double circle mesh")

    plt.tight_layout()


def test_discrete_phantom_uniform(size=100, ratio=9):
    """The uniform discrete phantom is the same after rotating 90 degrees."""

    d0 = discrete_phantom(p, size, ratio=ratio, prop='mass_attenuation')

    p.rotate(theta=np.pi/2, point=Point([0.5, 0.5]))
    d1 = np.rot90(discrete_phantom(p, size, ratio=ratio,
                                   prop='mass_attenuation'))

    # plot rotated phantom
    plot_phantom(p)
    plt.suptitle('rotated phantom')

    # plot the error
    plt.figure()
    plt.imshow(d1-d0, interpolation=None)
    plt.colorbar()
    plt.suptitle('test_discrete_phantom_uniform')

    # plt.show(block=True)
    # assert_allclose(d0, d1)


if __name__ == '__main__':
    test_plot_phantom_plain()
    test_plot_phantom_color_map()
    test_sidebyside()
    test_discrete_geometry()
    test_discrete_phantom_uniform()

    plt.show(block=True)

# def test_discrete_phantom_gaussian():
#     """Tests if the gaussian discrete phantom is the same after rotating the
#     phantom 90 degrees.
#     """
#     d0 = discrete_phantom(p, 100, ratio=10, uniform=False, prop='mass_attenuation')
#
#     p.rotate(np.pi/2)
#     d1 = np.rot90(discrete_phantom(p, 100, ratio=10, uniform=False,
#                   prop='mass_attenuation'))
#
#     # plot the error
#     plt.figure()
#     plt.imshow(d1-d0, interpolation=None)
#     plt.colorbar()
#
#     # plt.show(block=True)
#     assert_array_almost_equal(d0, d1)
