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
import string

logger = logging.getLogger(__name__)


__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['plot_phantom',
           'plot_metrics',
           'plot_histograms']


def plot_phantom(phantom):
    """Plots the phantom.

    Parameters
    ----------
    phantom : Phantom
    """
    fig = plt.figure(figsize=(8, 8), facecolor='w')
    a = fig.add_subplot(111, aspect='equal')

    # Draw all circles in the phantom.
    for m in range(phantom.population):
        cx = phantom.feature[m].center.x
        cy = phantom.feature[m].center.y
        cr = phantom.feature[m].radius
        circle = patches.Circle((cx, cy), cr)
        a.add_patch(circle)

    plt.grid('on')
    plt.gca().invert_yaxis()
    plt.show(block=False)


def plot_histograms(images, masks=None, thresh=0.025):
    """Plots the normalized histograms for the pixel intensity under each
    mask.

    Parameters
    --------------
    images : list of ndarrays, ndarray
        image(s) for comparing histograms.
    masks : list of ndarrays, float, optional
        If supplied, the data under each mask is plotted separately.
    strict : boolean
        If true, the mask takes values >= only. If false, the mask takes all
        values > 0.
    """
    if type(images) is not list:
        images = [images]

    hgrams = []  # holds histograms before plotting
    labels = []  # holds legend labels for plotting
    abet = string.ascii_uppercase

    if masks == None:
        for i in range(len(images)):
            hgrams.append(images[i])
            labels.append(abet[i])
    else:
        for i in range(len(masks)):
            for j in range(len(images)):
                m = masks[i]
                A = images[j]
                assert(A.shape == m.shape)
                # convert probability mask to boolean mask
                mA = A[m >= thresh]
                #h = np.histogram(m, bins='auto', density=True)
                hgrams.append(mA)
                labels.append(abet[j] + str(i))

    plt.figure()
    # autobins feature doesn't work because one of the groups is all zeros?
    plt.hist(hgrams, bins=25, normed=True, stacked=False)
    plt.legend(labels)
    plt.show()


def plot_metrics(imqual):
    """Plots metrics of ImageQuality data

    """
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(imqual))))
    for i in range(0, len(imqual)):
        # Draw a plot of the mean quality vs scale using different colors for
        # each reconstruction.
        plt.figure(0)
        plt.scatter(imqual[i].scales, imqual[i].qualities, color=next(colors))

        # Plot the reconstruction
        f = plt.figure(i + 1)
        N = len(imqual[i].maps) + 1
        p = _pyramid(N)
        plt.subplot2grid((p[0][0], p[0][0]), p[0][1], colspan=p[0][2],
                         rowspan=p[0][2])
        plt.imshow(imqual[i].recon, cmap=plt.cm.inferno,
                   interpolation="none", aspect='equal')
        plt.colorbar()
        plt.title("Reconstruction")

        lo = 1.  # Determine the min local quality for all the scales
        for m in imqual[i].maps:
            lo = min(lo, np.min(m))

        # Draw a plot of the local quality at each scale.
        for j in range(1, N):
            plt.subplot2grid((p[j][0], p[j][0]), p[j][1], colspan=p[j][2],
                             rowspan=p[j][2])
            im = plt.imshow(imqual[i].maps[j - 1], cmap=plt.cm.viridis, vmin=lo,
                            vmax=1, interpolation="none", aspect='equal')
            # plt.colorbar()
            plt.annotate(r'$\sigma$ =' + str(imqual[i].scales[j - 1]),
                         xy=(0.05, 0.05), xycoords='axes fraction',
                         weight='heavy')

        # plot one colorbar to the right of these images.
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im, cax=cbar_ax)
        plt.title(imqual[i].method)

        '''
        plt.subplot(121)
        plt.imshow(imqual[i].orig, cmap=plt.cm.viridis, vmin=0, vmax=1,
                   interpolation="none", aspect='equal')
        plt.title("Ideal")
        '''
    plt.figure(0)
    plt.ylabel('Quality')
    plt.xlabel('Scale')
    plt.ylim([0, 1])
    plt.legend([str(x) for x in range(1, len(imqual) + 1)])
    plt.title("Comparison of Reconstruction Methods")

    plt.show(block=True)


def _pyramid(N):
    """Generates the corner positions, grid size, and column/row spans for a pyramid image.

    Parameters
    --------------
    N : int
        the total number of images in the pyramid.

    Returns
    -------------
    params : list of lists
        Contains the params for subplot2grid for each of the N images in the
        pyramid. [W,corner,span] W is the total grid size, corner is the
        location of a particular axies, and span is the size of a paricular
        axies.
    """
    L = round(N / float(3))  # the number of levels in the pyramid
    W = int(2**L)  # grid size of the pyramid

    params = [p % 3 for p in range(0, N)]
    lcorner = [0, 0]  # the min corner of this level
    for n in range(0, N):
        l = int(n / 3)  # pyramid level
        span = int(W / (2**(l + 1)))  # span of the in number of grid spaces
        corner = list(lcorner)  # the min corner of this tile

        if params[n] == 0:
            lcorner[0] += span
            lcorner[1] += span
        elif params[n] == 2:
            corner[0] = lcorner[0] - span
        elif params[n] == 1:
            corner[1] = lcorner[1] - span

        params[n] = [W, corner, span]
        # print(params[n])

    return params
