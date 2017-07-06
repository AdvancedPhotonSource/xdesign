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

"""Contains functions for visualizing :class:`.Phantom` and
:class:`.ImageQuality` metrics.

.. moduleauthor:: Daniel J Ching <carterbox@users.noreply.github.com>
"""

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
import polytope as pt

logger = logging.getLogger(__name__)


__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['plot_phantom',
           'plot_geometry',
           'plot_mesh',
           'plot_polygon',
           'plot_curve',
           'discrete_phantom',
           'combine_grid',
           'discrete_geometry',
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
DEFAULT_ENERGY = 15

# cycle through 126 unique line styles
PLOT_STYLES = (14 * cycler('color', ['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']) +
               63 * cycler('linestyle', ['-', '--']) +
               18 * cycler('marker', ['o', 's', '.', 'D', '^', '*', '8']))


def plot_phantom(phantom, axis=None, labels=None, c_props=[], c_map=None, i=-1,
                 z=0.0, t=0.0001):
    """Plots a :class:`.Phantom` to the given axis.

    Parameters
    ----------
    phantom : :class:`.Phantom`
        A phantom to be plotted.
    axis : :class:`matplotlib.axis.Axis`
        The axis where the phantom should be plotted. `None` creates
        a new axis.
    labels : bool, optional
        `True` : Each :class:`.Phantom` given a unique number.
    c_props : list of str, optional
        List of :class:`.Phantom` properties to use for colormapping the
        geometries. `[]` colors the geometries by type.
    c_map : function, optional
        A function which takes the list of prop(s) for a :class:`.Phantom` as
        input and returns a matplolib color specifier. :cite:`Hunter:07`
    """
    if axis is None:
        fig, axis = _make_axis()
    if c_props and c_map is None:
        c_map = DEFAULT_COLOR_MAP

    props = list(c_props)
    num_props = range(0, len(c_props))

    if phantom.geometry is None:
        # can't plot without geometry. plot nothing
        pass
    else:
        plotted = False
        if phantom.material is None:
            # phantom has no properties. it is a container
            pass
        else:
            # have material and geometry...
            if c_map is None:
                # but no color assignments. plot default colors
                color = None
            else:
                # use the colormap to determine the color
                # TODO: Add parameter to pass other things besides energy
                for j in num_props:
                    props[j] = getattr(phantom.material, c_props[j])(energy=DEFAULT_ENERGY)
                color = c_map(props)[0]

            plotted = plot_geometry(phantom.geometry, axis, c=color, z=z, t=t)
            i += 1

        if plotted is not False and labels is not None:
            axis.annotate(str(i), xy=(phantom.geometry.center.x,
                                      phantom.geometry.center.y),
                          ha='center', va='center', color=LABEL_COLOR,
                          path_effects=[PathEffects.withStroke(
                            linewidth=3, foreground=DEFAULT_EDGE_COLOR)])

    for child in phantom.children:
        i = plot_phantom(child, axis=axis, labels=labels, c_props=c_props,
                         c_map=c_map, i=i, z=z, t=t)

    return i


def plot_geometry(geometry, axis=None, alpha=None, c=None, z=0.0, t=0.0001):
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
        return False
    elif isinstance(geometry, Mesh):
        return plot_mesh(geometry, axis, alpha, c)
    elif isinstance(geometry, Curve):
        return plot_curve(geometry, axis, alpha, c)
    elif isinstance(geometry, Polygon):
        return plot_polygon(geometry, axis, alpha, c)
    elif isinstance(geometry, pt.Polytope):
        return plot_polytope(geometry, axis, alpha, c, z, t)
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
        plot_geometry(f, axis, alpha, c)


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


def plot_polytope(polytope, axis=None, alpha=None, c=None, z=0.0, t=0.0001):
    """Project and plot a polytope into the plane"""
    if c is None:
        c = POLY_COLOR

    box_zmin = polytope.bounding_box[0][2]
    box_zmax = polytope.bounding_box[1][2]

    if (box_zmin < z and box_zmax < z
            or z + t < box_zmin and z + t < box_zmax):
        return False

    lo = [0, 0, z]
    hi = [1, 1, z + t]

    plane = pt.Polytope.from_box(np.stack([lo, hi], axis=1))

    projection = plane.intersect(polytope).project([1, 2])

    if projection.dim > 0:
        projection.plot(ax=axis, alpha=alpha, color=c)
        return True

    return False


def plot_curve(curve, axis=None, alpha=None, c=None):
    """Plots a :class:`.Curve` to the given axis.

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
    fig = plt.figure(figsize=(8, 8), dpi=100)
    axis = fig.add_subplot(111, aspect='equal')
    plt.grid('on')
    plt.gca().invert_yaxis()
    return fig, axis


def discrete_phantom(phantom, psize, bounding_box=((0, 0), (1, 1)),
                     ratio=9, uniform=True,
                     prop='linear_attenuation', energy=DEFAULT_ENERGY,
                     overlay_mode='add', return_patch=False):
    """Return a discrete map of the `property` in the `phantom`.

    The values of overlapping :class:`phantom.Phantom` are additive.

    Parameters
    ----------
    phantom: :class:`phantom.Phantom`
    psize : float
        The side length of a cubic pixel.
    bounding_box : :py:class:`numpy.ndarray`, :py:class:`List`
        The desired field of view specified as `[min, max]` where min and max
        are two column vectors pointing to the min and max corners of the
        desired field of view.
    ratio : scalar, optional (default: 9)
        The antialiasing works by supersampling. This parameter controls
        how many pixels in the larger representation are averaged for the
        final representation. e.g. if ratio = 9, then the final pixel
        values are the average of 81 pixels.
    uniform : boolean, optional (default: True)
        When set to False, changes the way pixels are averaged from a
        uniform weights to gaussian weigths.
    prop : str or list of str, optional (default: linear_attenuation)
        The name of the property to discretize
    fix_psize : bool
        Select whether use a fixed pixel size (at 1) or fixed object size
        (1cm)
    overlay_mode : 'add' or 'replace'
        Select the mode to overlay child phantom values to the existing grid.
        'add': values will be added
        'replace': parent values will be replaced by child values
    return_patch : bool
        Whether to return the combined patch of the current phantom and all its children.

    Return
    ------
    image : :class:`numpy.ndarray`
        The discrete representation of the :class:`.Phantom` that is size x
        size. 0 if phantom has no geometry or material property.

    Raise
    -----
    ValueError
        If size is less than or equal to 0
    """
    if psize <= 0:
        raise ValueError('size must be greater than 0.')

    bounding_box = np.asarray(bounding_box)
    if bounding_box.min() < 0 or bounding_box.max() > 1:
        raise ValueError('bounding box must be within (0, 1).')

    size = (bounding_box[1] - bounding_box[0]) / psize
    size = size.astype('int')

    image = ret = patch = 0

    prop = np.atleast_1d(prop)
    if phantom.geometry is not None and phantom.material is not None \
       and np.alltrue([hasattr(phantom.material, temp) for temp in prop]):
        # Rasterize all geometry in the phantom.
        pmin, patch = discrete_geometry(phantom.geometry, psize, ratio)

        # Get the property value
        n_prop = prop.size
        ret = [None] * n_prop
        for i, i_prop in enumerate(prop):
            try:
                value = getattr(phantom.material, i_prop)(energy=energy)
            except:
                raise TypeError('Invalid material properties.')

            # Make a grid to put store all of the discrete geometries
            image = np.zeros(size, dtype=float)
            imin = [0] * phantom.geometry.dim

            image = combine_grid(imin, image, pmin // psize, patch * value)
            ret[i] = image

        patch = patch.astype('bool')
        imin = [0] * phantom.geometry.dim
        patch = combine_grid(imin, np.zeros_like(image).astype('bool'), pmin // psize, patch)

    for child in phantom.children:
        temp, temp_patch = discrete_phantom(child, psize, bounding_box, ratio, uniform,
                                            prop, energy, return_patch=True, overlay_mode=overlay_mode)
        if ret is 0:
            ret = temp
        else:
            for i in range(len(ret)):
                if overlay_mode == 'add':
                    ret[i] = ret[i] + temp[i]
                elif overlay_mode == 'replace':
                    ret[i][temp_patch] = temp[i][temp_patch]
    # if ret is not 0:
    #     ret = np.squeeze(ret)
    if return_patch:
        try:
            patch += temp_patch
        except:
            pass
        return ret, patch
    else:
        return np.squeeze(ret)


def combine_grid(Amin, A, Bmin, B):
    """Add grid B to grid A by aligning min corners and clipping B

    Parameters
    ----------
    Amin, Bmin : int tuple
        The coordinates of the minimum corner of A and B
    A, B : numpy.ndarray
        The two arrays to add to each other

    Return
    ------
    AB : numpy.ndarray
        The combined grid

    Raise
    -----
    ValueError
        If A and B are do not have the same number of dimensions
    """
    if A.ndim != B.ndim:
        raise ValueError("A and B must have the same number of dimensions.")

    Amin = np.array(Amin, dtype=int)
    Bmin = np.array(Bmin, dtype=int)

    Amax = np.array(A.shape) + Amin
    Bmax = np.array(B.shape) + Bmin

    if np.any(Bmax <= Amin) or np.any(Amax <= Bmin):
        # B doesn't overlap A
        return A

    # for each dimension, crop and pad B to fit inside A

    forecrop = np.atleast_1d(Amin - Bmin)
    postcrop = np.atleast_1d(Amax - Bmax)

    pads = np.zeros([A.ndim, 2], dtype=int)
    for i in range(A.ndim):
        if forecrop[i] > 0:
            B = B[forecrop[i]:]
        if postcrop[i] < 0:
            B = B[:postcrop[i]]

        pads[0] = 0

        if forecrop[i] < 0:
            pads[0, 0] = -forecrop[i]
        if postcrop[i] > 0:
            pads[0, 1] = postcrop[i]

        B = np.pad(B, pads, 'constant')

        B = np.moveaxis(B, 0, -1)

    assert B.shape == A.shape, ("A:{} is not the same shape as "
                                "B:{}").format(A.shape, B.shape)

    return A + B


def discrete_geometry(geometry, psize, ratio=9):
    """Draw the geometry onto a patch the size of its bounding box.

    Parameters
    ----------
    geometry : :class:`geometry.Entity`
        A geometric object with `dim`, `bounding_box`, and `contains` methods
    psize : float [cm]
        The real size of the pixels in the discrete image
    ratio : int (default: 9)
        The supersampling ratio for antialiasing. 1 means no antialiasing

    Return
    ------
    corner : 1darray [cm]
        The min corner of the patch
    patch : ndarray
        The discretized geometry in it's bounding box

    Raise
    -----
    ValueError
        If `ratio` is less than 1 or `psize` is less than or equal to 0.
    """
    if ratio < 1:
        raise ValueError('ratio must be at least 1.')
    if ratio <= 0:
        raise ValueError('psize must be more than 0.')

    logger.debug("geometry: {}".format(repr(geometry)))

    # Determine the coordinates of the middle of each pixel in the supersampled
    # bounding box
    xmin, xmax = geometry.bounding_box
    imin, imax = xmin // psize, xmax // psize + 1

    margin = max(1, ratio // 2)  # buffer for rounding errors
    nsteps = imax - imin + 2 * margin

    # print(imin, imax, nsteps)

    pixel_coords = [None] * geometry.dim
    final_shape = np.zeros(geometry.dim, dtype=int)
    corner = np.zeros(geometry.dim)

    for i in range(geometry.dim):
        x = psize * ((imin.flat[i] - margin)
                     + np.arange(nsteps.flat[i] * ratio) / ratio)
        # TODO: @carterbox Determine whether arange, or linspace works better
        # at surpressing rotation error. SEE test_discrete_phantom_uniform

        # Check whether the patch range, x, contains the bounding box
        assert x[0] <= xmin.flat[i], x[0]
        assert xmax.flat[i] < x[-1] + psize / ratio, x[-1] + psize / ratio
        # The length of x should be an integer multiple of the decimation ratio
        assert x.size % ratio == 0, x.size

        corner[i] = x[0]

        x += psize / (2 * ratio)  # move point to mid-pixel

        pixel_coords[i] = x
        final_shape[i] = x.size

    # Reshape the pixels_coords into an MxN array
    pixel_coords = np.stack(np.meshgrid(*pixel_coords, indexing='ij'), axis=-1)
    pixel_coords = np.reshape(pixel_coords, (np.prod(pixel_coords.shape[0:-1]),
                                             geometry.dim))

    # Compute whether each pixel is contained within the geometry
    image = geometry.contains(pixel_coords)
    image.shape = final_shape
    image = image.astype(float)

    # Resample down to the desired size.
    if True:
        image = scipy.ndimage.uniform_filter(image, ratio, mode='constant')
    else:
        image = scipy.ndimage.gaussian_filter(image, np.sqrt(ratio/2))

    # Roll image so that decimation chooses
    # from the exact center of each filter when ratio is odd.
    patch = multiroll(image, [-ratio//2 + 1]*geometry.dim)

    # Decimate each axis
    for i in range(geometry.dim):
        patch = patch[::ratio]
        patch = np.moveaxis(patch, 0, -1)

    # Check that the resulting image is the expected size
    assert np.all(patch.shape == final_shape // ratio)

    if geometry.dim > 1:
        patch = np.swapaxes(patch, 0, 1)
        corner[0], corner[1] = corner[1], corner[0]

    # Return the image and its min corner
    return corner, patch


def sidebyside(p, size=100, labels=None, prop='mass_attenuation'):
    '''Displays the geometry and the discrete property function of
    the given :class:`.Phantom` side by side.'''
    # plt.rcParams.update({'font.size': 6})

    fig = plt.figure(figsize=(6, 3), dpi=100)

    axis = fig.add_subplot(121, aspect='equal')
    plot_phantom(p, axis=axis, labels=labels)
    plt.grid('on')
    axis.invert_yaxis()
    axis.set_xticks(np.linspace(0, 1, 6, True))
    axis.set_yticks(np.linspace(0, 1, 6, True))

    axis = plt.subplot(122)
    d = discrete_phantom(p, 1. / size, prop=prop)
    plt.imshow(d, interpolation='none', cmap=plt.cm.inferno)
    axis.set_xticks(np.linspace(0, size, 6, True))
    axis.set_yticks(np.linspace(0, size, 6, True))

    plt.tight_layout()

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
    """Plot local quality maps from ImageQuality data.

    Parameters
    ----------
    imqual : ImageQuality
        The data to plot.

    References
    ----------
    Colors taken from this gist <https://gist.github.com/thriveth/8560036>
    """

    # Plot the reconstruction
    f = plt.figure()
    N = len(imqual.maps) + 1
    p = _pyramid(N)
    plt.subplot2grid((p[0][0], p[0][0]), p[0][1], colspan=p[0][2],
                     rowspan=p[0][2])
    plt.imshow(imqual.img1, cmap=plt.cm.inferno,
               interpolation="none", aspect='equal')
    # plt.colorbar()
    plt.axis('off')
    # plt.title("Reconstruction")

    lo = 1.  # Determine the min local quality for all the scales
    for m in imqual.maps:
        lo = min(lo, np.min(m))

    # Draw a plot of the local quality at each scale.
    for j in range(1, N):
        plt.subplot2grid((p[j][0], p[j][0]), p[j][1], colspan=p[j][2],
                         rowspan=p[j][2])
        im = plt.imshow(imqual.maps[j - 1], cmap=plt.cm.viridis,
                        vmin=lo, vmax=1, interpolation="none",
                        aspect='equal')
        # plt.colorbar()
        plt.axis('off')
        plt.annotate(r'$\sigma$ =' + str(imqual.scales[j - 1]),
                     xy=(0.05, 0.05), xycoords='axes fraction',
                     weight='heavy')

    # plot one colorbar to the right of these images.
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im, cax=cbar_ax)
    plt.title(imqual.method)

    '''
    plt.subplot(121)
    plt.imshow(imqual.orig, cmap=plt.cm.viridis, vmin=0, vmax=1,
               interpolation="none", aspect='equal')
    plt.title("Ideal")
    '''


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


def plot_wavefront(simulator):
    """Plot wavefront intensity.

    Parameters:
    -----------
    simulator : :class:`Simulator`
    """
    wavefront = simulator.wavefront
    i = np.abs(wavefront * np.conjugate(wavefront))

    fig = plt.figure()
    plt.imshow(i, cmap='gray')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
