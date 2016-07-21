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
from phantom.geometry import *
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['sinogram',
           'angleogram']


def sinogram(sx, sy, phantom):
    """Generates sinogram given a phantom.

    Parameters
    ----------
    sx : int
        Number of rotation angles.
    sy : int 
        Number of detection pixels (or sample translations).
    phantom : Phantom 
    """
    # Step size of the probe.
    size = 1. / sy

    # Step size of the rotation angle.
    theta = np.pi / sx

    # Fixed probe location.
    p = Probe(Point(size / 2., 0), Point(size / 2., 1), size)

    # Measure sinogram values.
    sino = np.zeros((sx, sy))
    for m in range(sx):
        # print ("theta=%s" % (m * theta * 180. / np.pi))
        for n in range(sy):
            sino[m, n] = p.measure(phantom)
            phantom.translate(-size, 0)
        phantom.translate(1, 0)
        phantom.rotate(-theta, Point(0.5, 0.5))
    return sino


def angleogram(sx, sy, phantom):
    """Generates angleogram given a phantom.

    Parameters
    ----------
    sx : int
        Number of rotation angles.
    sy : int 
        Number of detection pixels (or sample translations).
    phantom : Phantom
    """
    # Step size of the probe.
    size = 1. / sy

    # Fixed rotation points.
    p1 = Point(0.0, 0.5)
    p2 = Point(0.5, 0.5)

    # Step sizes of the rotation angles.
    alpha = np.pi / sx
    beta = np.pi / sy
    print (alpha, beta)

    # Fixed probe location.
    p = Probe(Point(size / 2., 0), Point(size / 2., 1), size)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Measure angleogram values.
    angl = np.zeros((sx, sy))
    for m in range(sx):
        print ("alpha=%s" % (m * alpha * 180. / np.pi))
        for n in range(sy):
            angl[m, n] = p.measure(phantom)
            phantom.rotate(alpha, p1)
        phantom.rotate(-np.pi, p1)
        phantom.rotate(beta, p2)
    return angl
