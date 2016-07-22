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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import time

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['plot_phantom']


def plot_phantom(phantom):
    """Plots the phantom.

    Parameters
    ----------
    phantom : Phantom
    """
    fig = plt.figure(figsize=(8, 8), facecolor='w')
    a = fig.add_subplot(111)

    # Draw all circles in the phantom.
    for m in range(phantom.population):
        cx = phantom.feature[m].center.x
        cy = phantom.feature[m].center.y
        cr = phantom.feature[m].radius
        circle = patches.Circle((cx, cy), cr)
        a.add_patch(circle)

    plt.grid('on')
    plt.show()

def plot_metrics(imqual):
    """Plots metrics of ImageQuality data

    """
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(imqual))))
    for i in range(0,len(imqual)):
        # Draw a plot of the mean quality vs scale using different colors for each reconstruction
        plt.figure(0)
        plt.scatter(imqual[i].scales, imqual[i].qualities,color=next(colors))

        # Draw a plot of the local quality at each scale on the same figure as the original
        plt.figure(i+1)
        N = 2
        plt.subplot2grid((N,N),(0,0),colspan=1,rowspan=1)
        plt.imshow(imqual[i].recon, cmap=plt.cm.gray)
        plt.subplot2grid((N,N),(1,0),colspan=1,rowspan=1)
        plt.imshow(imqual[i].maps[0], cmap=plt.cm.gray)

    plt.figure(0)
    plt.ylabel('Quality')
    plt.xlabel('Scale')
    plt.legend([str(x) for x in range(1,len(imqual)+1)])

    plt.show(block=False)
    time.sleep(5)
    plt.close("all")
