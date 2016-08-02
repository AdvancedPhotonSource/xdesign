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
from phantom.geometry import *

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Phantom']


class Phantom(object):
    """Phantom generation class.

    Attributes
    ----------
    shape : string
        Shape of the phantom. Available options: circle, square.
    population : scalar
        Number of generated features in the phantom.
    area : scalar
        Area of the features in the phantom.
    feature : list
        List of features.
    """

    def __init__(self, shape='circle'):
        self.shape = shape
        self.population = 0
        self.area = 0
        self.feature = []

    @property
    def list(self):
        for m in range(self.population):
            print (self.feature[m].list)

    @property
    def density(self):
        if self.shape == 'square':
            return self.area
        elif self.shape == 'circle':
            return self.area / (np.pi * 0.5 * 0.5)

    def add(self, feature):
        """Add a feature to the phantom.

        Parameters
        ----------
        feature : Feature
        """
        self.feature.append(feature)
        self.area += feature.area
        self.population += 1

    def pop(self):
        """Pop last added feature from the phantom."""
        self.population -= 1
        self.area -= self.feature[-1].area
        self.feature.pop()

    def _random_point(self, margin=0):
        """Generate a random point in the phantom.

        Parameters
        ----------
        margin : scalar
            Determines the margin value of the shape.
            Points will not be created in the margin area.

        Returns
        -------
        Point
            Random point.
        """
        if self.shape == 'square':
            x = np.random.uniform(margin, 1 - margin)
            y = np.random.uniform(margin, 1 - margin)
        elif self.shape == 'circle':
            r = np.random.uniform(0, 0.5 - margin)
            a = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(a) + 0.5
            y = r * np.sin(a) + 0.5
        return Point(x, y)

    def sprinkle(self, counts, radius, gap=0, collision=False):
        """Sprinkle a number of circles.

        Parameters
        ----------
        counts : int
            Number of circles to be added.
        gap : float, optional
            Minimum distance between circle boundaries.
        collision : bool, optional
            True if circles are allowed to overlap.
        """
        for m in range(int(counts)):
            center = self._random_point(radius)
            circle = Circle(center, radius + gap)
            if not self.collision(circle) or collision:
                self.add(Circle(center, radius))

    def collision(self, circle):
        """Check if a circle is collided with another circle."""
        for m in range(self.population):
            dx = self.feature[m].center.x - circle.center.x
            dy = self.feature[m].center.y - circle.center.y
            dr = self.feature[m].radius + circle.radius
            if np.sqrt(dx**2 + dy**2) < dr:
                return True
        return False

    def numpy(self):
        """Returns the Numpy representation."""
        arr = np.empty((self.population, 4))
        for m in range(self.population):
            arr[m] = [
                self.feature[m].center.x,
                self.feature[m].center.y,
                self.feature[m].radius,
                self.feature[m].value]
        return arr

    def save(self, filename):
        """Save phantom to file."""
        np.savetxt(filename, self.numpy(), delimiter=',')

    def load(self, filename):
        """Load phantom from file."""
        arr = np.loadtxt(filename, delimiter=',')
        for m in range(arr.shape[0]):
            self.add(Circle(Point(arr[m, 0], arr[m, 1]), arr[m, 2], arr[m, 3]))

    def translate(self, dx, dy):
        """Translate phantom."""
        for m in range(self.population):
            self.feature[m].translate(dx, dy)

    def rotate(self, theta, origin=Point(0, 0)):
        """Rotate phantom around a point."""
        for m in range(self.population):
            self.feature[m].rotate(theta, origin)

    def discrete(self, size, bitdepth=8, ratio=8, uniform=True):
        """Returns discrete representation of the phantom.

        Parameters
        ------------
        size : scalar
            The side length in pixels of the resulting square image.
        bitdepth : scalar, optional
            The bitdepth of resulting representation. Depths less than 32 are 
            returned as integers, and depths greater than 32 are returned as 
            floats.
        ratio : scalar, optional
            The discretization works by drawing the shapes in a larger space 
            then averaging and downsampling. This parameter controls how many 
            pixels in the larger representation are averaged for the final 
            representation. e.g. if ratio = 8, then the final pixel values 
            are the average of 64 pixels.
        uniform : boolean, optional
            When set to False, changes the way pixels are averaged from a 
            uniform weights to gaussian weights.
        
        Returns
        ------------
        image : numpy.ndarray
            The discrete representation of the phantom.
        """
        # Make a higher resolution grid to sample the continuous space
        _x = np.arange(0, 1, 1 / size / ratio)
        _y = np.arange(0, 1, 1 / size / ratio)
        px, py = np.meshgrid(_x, _y)
        
        # Draw the shapes at the higher resolution
        image = np.zeros((size*ratio,size*ratio), dtype=np.float)
        for m in range(self.population):
            x = self.feature[m].center.x
            y = self.feature[m].center.y
            rad = self.feature[m].radius
            val = self.feature[m].value
            image += ((px - x)**2 + (py - y)**2 < rad**2) * val
                
#        import matplotlib.pylab as plt
#        plt.imshow(image, cmap=plt.cm.viridis)
#        plt.colorbar()
#        plt.show(block=False)
        
        # Resample down to the desired size
        if uniform:
            image = scipy.ndimage.uniform_filter(image,ratio)
        else:
            image = scipy.ndimage.gaussian_filter(image,ratio/4.)
        image = image[::ratio,::ratio]

#        print(image.shape)
#        print(np.max(image))

        # Rescale to proper bitdepth
        if bitdepth < 32:
            image = image*(2**bitdepth-1)
            image = image.astype(int)
            
#        plt.figure()
#        plt.imshow(image, cmap=plt.cm.viridis)
#        plt.colorbar()
#        plt.show()
        return image
