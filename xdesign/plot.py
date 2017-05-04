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

import logging
import types
import time
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
import scipy.ndimage
from cycler import cycler
from xdesign.phantom import Phantom
from xdesign.geometry import Curve, Polygon, Mesh
from matplotlib.axis import Axis
from itertools import product
from six import string_types

logger = logging.getLogger(__name__)


__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['plot_phantom',
           'plot_mesh',
           'plot_polygon',
           'plot_curve',
           'discrete_phantom',
           'sidebyside',
           'multiroll',
           'plot_metrics',
           'plot_mtf',
           'plot_nps',
           'plot_neq',
           'plot_histograms']

DEFAULT_COLOR_MAP = plt.cm.viridis
DEFAULT_COLOR = DEFAULT_COLOR_MAP(0.25)
POLY_COLOR = DEFAULT_COLOR_MAP(0.8)
DEFAULT_EDGE_COLOR = 'white'
POLY_EDGE_COLOR = 'black'
LABEL_COLOR = 'black'
POLY_LINEWIDTH = 0.1
CURVE_LINEWIDTH = 0.5

# cycle through 126 unique line styles
PLOT_STYLES = (14 * cycler('color', ['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']) +
               63 * cycler('linestyle', ['-', '--']) +
               18 * cycler('marker', ['o', 's', '.', 'D', '^', '*', '8']))


def plot_phantom(phantom, axis=None, labels=None, c_props=[], c_map=None, i=0):
    """Plots a :class:`.Phantom` to the given axis.

    Parameters
    ----------
    phantom : :class:`.Phantom`
        A phantom to be plotted.
    axis : :class:`matplotlib.axis.Axis`
        The axis where the phantom should be plotted. `None` creates
        a new axis.
    labels : bool, optional
        `True` : Each :class:`.Phantom` given a uniqe number.
    c_props : list of str, optional
        List of :class:`.Phantom` properties to use for colormapping the
        geometries.
    c_map : function, optional
        A function which takes the list of prop(s) for a :class:`.Phantom` as
        input and returns a matplolib color specifier. :cite:`Hunter:07`
    """
    assert isinstance(phantom, Phantom), ('phantom is a ' +
                                          '{}'.format(type(phantom)))

    # IDEA: Allow users to provide list or generator for labels.
    if axis is None:
        fig, axis = _make_axis()
    if not isinstance(c_props, list):
        raise TypeError('c_props must be list of str')
    if len(c_props) > 0 and c_map is None:
        c_map = DEFAULT_COLOR_MAP

    props = list(c_props)
    num_props = range(0, len(c_props))

    # Draw geometry in the phantom.
    if phantom.geometry is not None:
        if c_map is not None:
            # use the colormap to determine the color
            for j in num_props:
                props[j] = getattr(phantom, c_props[j])
            color = c_map(props)[0]
        else:
            color = None

        plot_geometry(phantom.geometry, axis, c=color)
        if labels is not None:
            axis.annotate(str(i), xy=(phantom.geometry.center.x,
                                      phantom.geometry.center.y),
                          ha='center', va='center', color=LABEL_COLOR,
                          path_effects=[PathEffects.withStroke(
                            linewidth=3, foreground=DEFAULT_EDGE_COLOR)])
            i += 1

    for child in phantom.children:
        i = plot_phantom(child, axis=axis, labels=labels, c_props=c_props,
                         c_map=c_map, i=i)

    return i


def plot_geometry(geometry, axis=None, alpha=None, c=None):
    """Plots a :class:`.Entity` on the given axis.

    Parameters
    ----------
    geometry : :class:`.Entity`
        A geometry to plot on the given axis.
    axis : :class:`matplotlib.axis.Axis`, optional
        The axis where the geometry should be plotted. `None` creates
        a new axis.
    alpha : :class:`.float`, optional
        The plot opaqueness. 0 is transparent. 1 is opaque.
    c : :mod:`matplotlib.color`, optional
        The color of the plotted geometry.
    """
    if axis is None:
        fig, axis = _make_axis()

    # Plot geometry using correct method
    if geometry is None:
        return
    elif isinstance(geometry, Mesh):
        plot_mesh(geometry, axis, alpha, c)
    elif isinstance(geometry, Curve):
        plot_curve(geometry, axis, alpha, c)
    elif isinstance(geometry, Polygon):
        plot_polygon(geometry, axis, alpha, c)
    else:
        raise NotImplemented('geometry is not Mesh, Curve or Polygon.')


def plot_mesh(mesh, axis=None, alpha=None, c=None):
    """Plots a :class:`.Mesh` to the given axis.

    Parameters
    ----------
    mesh : :class:`.Mesh`
        A Mesh to plot on the given axis.
    axis : :class:`matplotlib.axis.Axis`, optional
        The axis where the Mesh should be plotted. `None` creates
        a new axis.
    alpha : :class:`.float`, optional
        The plot opaqueness. 0 is transparent. 1 is opaque.
    c : :mod:`matplotlib.color`, optional
        The color of the plotted Mesh.
    """
    assert(isinstance(mesh, Mesh))
    if axis is None:
        fig, axis = _make_axis()

    # Plot each face separately
    for f in mesh.faces:
        plot_polygon(f, axis, alpha, c)


def plot_polygon(polygon, axis=None, alpha=None, c=None):
    """Plots a :class:`.Polygon` to the given axis.

    Parameters
    ----------
    polygon : :class:`.Polygon`
        A Polygon to plot on the given axis.
    axis : :class:`matplotlib.axis.Axis`, optional
        The axis where the Polygon should be plotted. `None` creates
        a new axis.
    alpha : :class:`.float`, optional
        The plot opaqueness. 0 is transparent. 1 is opaque.
    c : :mod:`matplotlib.color`, optional
        The color of the plotted Polygon.
    """
    assert(isinstance(polygon, Polygon))
    if axis is None:
        fig, axis = _make_axis()
    if c is None:
        c = POLY_COLOR

    p = polygon.patch
    p.set_alpha(alpha)
    p.set_facecolor(c)
    p.set_edgecolor(POLY_EDGE_COLOR)
    p.set_linewidth(POLY_LINEWIDTH)
    axis.add_patch(p)


def plot_curve(curve, axis=None, alpha=None, c=None):
    """Plots a :class:`.Curve' to the given axis.

    Parameters
    ----------
    curve : :class:`.Curve`
        A Curve to plot on the given axis.
    axis : :class:`matplotlib.axis.Axis`, optional
        The axis where the Curve should be plotted. None creates
        a new axis.
    alpha : :class:`.float`, optional
        The plot opaqueness. 0 is transparent. 1 is opaque.
    c : :mod:`matplotlib.color`, optional
        The color of the plotted curve.
    """
    assert(isinstance(curve, Curve))
    if axis is None:
        fig, axis = _make_axis()
    if c is None:
        c = DEFAULT_COLOR

    p = curve.patch
    p.set_alpha(alpha)
    p.set_facecolor(c)
    p.set_edgecolor(DEFAULT_EDGE_COLOR)
    p.set_linewidth(CURVE_LINEWIDTH)
    axis.add_patch(p)


def _make_axis():
    """Makes an :class:`matplotlib.axis.Axis` for plotting :mod:`.Phantom` module
    classes."""
    fig = plt.figure(figsize=(8, 8), facecolor='w')
    axis = fig.add_subplot(111, aspect='equal')
    plt.grid('on')
    plt.gca().invert_yaxis()
    return fig, axis


def discrete_phantom(phantom, size, ratio=8, uniform=True, prop='mass_atten'):
    """Returns discrete representation of the property function, prop, in the
    :class:`.Phantom`. The values of overlapping Phantoms are additive.

    Parameters
    ----------
    phantom: :class:`.Phantom`
    size : scalar
        The side length in pixels of the resulting square image.
    ratio : scalar, optional
        The antialiasing works by supersampling. This parameter controls
        how many pixels in the larger representation are averaged for the
        final representation. e.g. if ratio = 8, then the final pixel
        values are the average of 64 pixels.
    uniform : boolean, optional
        When set to False, changes the way pixels are averaged from a
        uniform weights to gaussian weigths.
    prop : str, optional
        The name of the property function to discretize

    Returns
    -------
    image : numpy.ndarray
        The discrete representation of the :class:`.Phantom` that is size x
        size.
    """
    if size <= 0:
        raise ValueError('size must be greater than 0.')
    if ratio < 1:
        raise ValueError('ratio must be at least 1.')
    ndims = 2

    # Make a higher resolution grid to sample the continuous space. Sample at
    # the center of each pixel.
    grid_step = 1 / size / ratio
    _x = np.arange(0, 1, grid_step) + grid_step / 2
    _y = np.arange(0, 1, grid_step) + grid_step / 2
    px, py = np.meshgrid(_x, _y)

    # Draw the shapes at the higher resolution.
    image = np.zeros((size * ratio, size * ratio), dtype=np.float)

    # Rasterize all geometry in the phantom.
    image = _discrete_geometry(phantom, image, px, py, prop)

    # Resample down to the desired size. Roll image so that decimation chooses
    # from the center of each pixel.
    if uniform:
        image = scipy.ndimage.uniform_filter(image, ratio)
    else:
        image = scipy.ndimage.gaussian_filter(image, np.sqrt(ratio/2))
    image = multiroll(image, [-ratio//2]*ndims)
    image = image[::ratio, ::ratio]

    assert(image.shape[0] == size and image.shape[1] == size)
    return image


def _discrete_geometry(phantom, image, px, py, prop):
    """Draw the geometry of the phantom onto the image.

    (px, py) are two arrays the same shape as image which hold the coordinates
    of image pixels. Multiply the geometry of each phantom by the value of
    phantom.prop.
    """
    if hasattr(phantom, prop) and phantom.geometry is not None:
        value = getattr(phantom, prop)

        size = px.shape  # is equivalent to image.shape?
        pixel_coords = np.vstack([px.flatten(), py.flatten()]).T

        logger.debug("pixel_coords: {}".format(pixel_coords))
        logger.debug("geometry: {}".format(phantom.geometry))

        new_feature = phantom.geometry.contains(pixel_coords) * value
        logger.debug("new_feature: {}".format(new_feature))

        new_feature = np.reshape(new_feature, size)

        image += new_feature

    for child in phantom.children:
        image = _discrete_geometry(child, image, px, py, prop)

    return image


def sidebyside(p, size=100, labels=None, prop='mass_atten'):
    '''Displays the geometry and the discrete property function of
    the given :class:`.Phantom` side by side.'''
    fig = plt.figure(figsize=(6, 3), dpi=600)
    axis = fig.add_subplot(121, aspect='equal')
    plt.grid('on')
    plt.gca().invert_yaxis()
    plot_phantom(p, axis=axis, labels=labels)
    plt.subplot(1, 2, 2)
    d = discrete_phantom(p, size, prop=prop)
    plt.imshow(d, interpolation='none', cmap=plt.cm.inferno)
    return d


def multiroll(x, shift, axis=None):
    """Roll an array along each axis.

    Parameters
    ----------
    x : array_like
        Array to be rolled.
    shift : sequence of int
        Number of indices by which to shift each axis.
    axis : sequence of int, optional
        The axes to be rolled.  If not given, all axes is assumed, and
        len(shift) must equal the number of dimensions of x.

    Returns
    -------
    y : numpy array, with the same type and size as x
        The rolled array.

    Notes
    -----
    The length of x along each axis must be positive.  The function
    does not handle arrays that have axes with length 0.

    See Also
    --------
    numpy.roll

    Example
    -------
    Here's a two-dimensional array:

    >>> x = np.arange(20).reshape(4,5)
    >>> x
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])

    Roll the first axis one step and the second axis three steps:

    >>> multiroll(x, [1, 3])
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])

    That's equivalent to:

    >>> np.roll(np.roll(x, 1, axis=0), 3, axis=1)
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])

    Not all the axes must be rolled.  The following uses
    the `axis` argument to roll just the second axis:

    >>> multiroll(x, [2], axis=[1])
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])

    which is equivalent to:

    >>> np.roll(x, 2, axis=1)
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])

    References
    ----------
    Warren Weckesser
    http://stackoverflow.com/questions/30639656/numpy-roll-in-several-dimensions
    """
    x = np.asarray(x)
    if axis is None:
        if len(shift) != x.ndim:
            raise ValueError("The array has %d axes, but len(shift) is only "
                             "%d. When 'axis' is not given, a shift must be "
                             "provided for all axes." % (x.ndim, len(shift)))
        axis = range(x.ndim)
    else:
        # axis does not have to contain all the axes.  Here we append the
        # missing axes to axis, and for each missing axis, append 0 to shift.
        missing_axes = set(range(x.ndim)) - set(axis)
        num_missing = len(missing_axes)
        axis = tuple(axis) + tuple(missing_axes)
        shift = tuple(shift) + (0,)*num_missing

    # Use mod to convert all shifts to be values between 0 and the length
    # of the corresponding axis.
    shift = [s % x.shape[ax] for s, ax in zip(shift, axis)]

    # Reorder the values in shift to correspond to axes 0, 1, ..., x.ndim-1.
    shift = np.take(shift, np.argsort(axis))

    # Create the output array, and copy the shifted blocks from x to y.
    y = np.empty_like(x)
    src_slices = [(slice(n-shft, n), slice(0, n-shft))
                  for shft, n in zip(shift, x.shape)]
    dst_slices = [(slice(0, shft), slice(shft, n))
                  for shft, n in zip(shift, x.shape)]
    src_blks = product(*src_slices)
    dst_blks = product(*dst_slices)
    for src_blk, dst_blk in zip(src_blks, dst_blks):
        y[dst_blk] = x[src_blk]

    return y


def plot_metrics(imqual):
    """Plots full reference metrics of ImageQuality data.

    Parameters
    ----------
    imqual : ImageQuality
        The data to plot.

    References
    ----------
    Colors taken from this gist <https://gist.github.com/thriveth/8560036>
    """
    fig_lineplot = plt.figure(0)
    plt.rc('axes', prop_cycle=PLOT_STYLES)

    for i in range(0, len(imqual)):
        # Draw a plot of the mean quality vs scale using different colors for
        # each reconstruction.
        plt.figure(fig_lineplot.number)
        plt.plot(imqual[i].scales, imqual[i].qualities)

        # Plot the reconstruction
        f = plt.figure(i + 1)
        N = len(imqual[i].maps) + 1
        p = _pyramid(N)
        plt.subplot2grid((p[0][0], p[0][0]), p[0][1], colspan=p[0][2],
                         rowspan=p[0][2])
        plt.imshow(imqual[i].recon, cmap=plt.cm.inferno,
                   interpolation="none", aspect='equal')
        # plt.colorbar()
        plt.axis('off')
        # plt.title("Reconstruction")

        lo = 1.  # Determine the min local quality for all the scales
        for m in imqual[i].maps:
            lo = min(lo, np.min(m))

        # Draw a plot of the local quality at each scale.
        for j in range(1, N):
            plt.subplot2grid((p[j][0], p[j][0]), p[j][1], colspan=p[j][2],
                             rowspan=p[j][2])
            im = plt.imshow(imqual[i].maps[j - 1], cmap=plt.cm.viridis,
                            vmin=lo, vmax=1, interpolation="none",
                            aspect='equal')
            # plt.colorbar()
            plt.axis('off')
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
    plt.figure(fig_lineplot.number)
    plt.ylabel('Quality')
    plt.xlabel('Scale')
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend([str(x) for x in range(1, len(imqual) + 1)])
    plt.title("Comparison of Reconstruction Methods")


def _pyramid(N):
    """Generates the corner positions, grid size, and column/row spans for
    a pyramid image.

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
    num_levels = round(N / float(3))  # the number of levels in the pyramid
    W = int(2**num_levels)  # grid size of the pyramid

    params = [p % 3 for p in range(0, N)]
    lcorner = [0, 0]  # the min corner of this level
    for n in range(0, N):
        level = int(n / 3)  # pyramid level
        span = int(W / (2**(level + 1)))  # span in num of grid spaces
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


def plot_mtf(faxis, MTF, labels=None):
    """Plots the MTF. Returns the figure reference."""
    fig_lineplot = plt.figure()
    plt.rc('axes', prop_cycle=PLOT_STYLES)

    for i in range(0, MTF.shape[0]):
        plt.plot(faxis, MTF[i, :])

    plt.xlabel('spatial frequency [cycles/length]')
    plt.ylabel('Radial MTF')
    plt.gca().set_ylim([0, 1])

    if labels is not None:
        plt.legend([str(n) for n in labels])
    plt.title("Modulation Tansfer Function for various angles")

    return fig_lineplot


def plot_nps(X, Y, NPS):
    """Plots the 2D frequency plot for the NPS.
    Returns the figure reference."""
    fig_nps = plt.figure()
    plt.contourf(X, Y, NPS, cmap='inferno')
    plt.xlabel('spatial frequency [cycles/length]')
    plt.ylabel('spatial frequency [cycles/length]')
    plt.axis(tight=True)
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.title('Noise Power Spectrum')
    return fig_nps


def plot_neq(freq, NEQ):
    """Plots the NEQ. Returns the figure reference."""
    fig_neq = plt.figure()
    plt.plot(freq.flatten(), NEQ.flatten())
    plt.xlabel('spatial frequency [cycles/length]')
    plt.title('Noise Equivalent Quanta')
    return fig_neq


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

    if masks is None:
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
                # h = np.histogram(m, bins='auto', density=True)
                hgrams.append(mA)
                labels.append(abet[j] + str(i))

    plt.figure()
    # autobins feature doesn't work because one of the groups is all zeros?
    plt.hist(hgrams, bins=25, normed=True, stacked=False)
    plt.legend(labels)
