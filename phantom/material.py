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
import logging
from phantom.phantom import *
from phantom.geometry import *

logger = logging.getLogger(__name__)

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['HyperbolicConcentric', 'DynamicRange','Soil', 'Foam', 'UnitCircle']

## Elements and Mixtures - Not Implemented
class Element(Circle):
    """
    """
    def __init__(self, MeV=1):
        # calculate the value based on the photon energy
        super(Element, self).__init__()

    @property
    def attenuation():
        raise NotImplementedError

    def set_beam_energy():
        raise NotImplementedError

## Microstructures
class HyperbolicConcentric(Phantom):
    """Generates a standard test pattern based on the ISO 12233:2014 standard.
    It is a series of cocentric alternating black and white circles whose radii
    are changing at a parabolic rate. These lines whose spacing covers a range
    scales can be used to quanitify the Modulation Transfer Function (MTF).
    """
    def __init__(self, min_width=0.1,exponent=1/2):
        """
        Attributes
        -------------
        radii : list
            The list of radii of the circles
        widths : list
            The list of the widths of the bands
        """
        super(HyperbolicConcentric, self).__init__(shape='circle')
        center = Point(0.5,0.5)
        #exponent = 1/2
        Nmax_rings = 512

        radii = [0]
        widths = [min_width]
        for ring in range(0,Nmax_rings):
            radius = min_width*np.power(ring+1,exponent)
            if radius > 0.5 and ring%2:
                break

            self.append(Circle(center,radius, value=(-1.)**(ring%2)))
            # record information about the rings
            widths.append(radius-radii[-1])
            radii.append(radius)

        self.reverse() # smaller circles on top
        self.radii = radii
        self.widths = widths

class DynamicRange(Phantom):
    """Generates a random placement of circles for determining dynamic range.
    """
    def __init__(self, steps=10, jitter=True, shape='square'):
        super(DynamicRange, self).__init__(shape=shape)

        # determine the size and and spacing of the circles around the box.
        spacing = 1./np.ceil(np.sqrt(steps))
        radius = spacing/4

        colors = [2**j for j in range(0,steps)]
        np.random.shuffle(colors)

        if jitter:
            # generate grid
            _x = np.arange(0, 1, spacing) + spacing/2
            px, py, = np.meshgrid(_x, _x)
            px = np.ravel(px)
            py = np.ravel(py)

            # calculate jitters
            jitters = 2*radius*(np.random.rand(2,steps)-0.5)

            # place the circles
            for i in range(0,steps):
                center = Point(px[i]+jitters[0,i],py[i]+jitters[1,i])
                self.append(Circle(center,radius,value=colors[i]))
        else:
            # completely random
            for i in range(0,steps):
                self.sprinkle(1,radius,gap=radius*0.9,value=colors[i])

class UnitCircle(Phantom):
    """Generates a phantom with a single circle of radius 0.4 for the purpose of
    measuring the nose power spectrum."""
    def __init__(self, value=1, radius=0.5):
        super(UnitCircle, self).__init__()
        self.append(Circle(Point(0.5,0.5),radius,value=value))

class Soil(Phantom):
    """Generates a phantom with structure similar to soil.

    References
    -----------
    Schlüter, S., Sheppard, A., Brown, K., & Wildenschild, D. (2014). Image
    processing of multiphase images obtained via X‐ray microtomography: a
    review. Water Resources Research, 50(4), 3615-3639.
    """
    def __init__(self, porosity=0.412):
        super(Soil, self).__init__(shape='circle')
        self.sprinkle(30, [0.1,0.03], 0, value=0.5, max_density=1-porosity)
        # use overlap to approximate area opening transform because opening is not discrete
        self.sprinkle(50, 0.02, 0.01, value=-.25)
        background = Circle(Point(0.5,0.5),0.5, value=0.5)
        self.insert(0,background)

class Foam(Phantom):
    """Generates a phantom with structure similar to foam."""
    def __init__(self):
        super(Foam, self).__init__(shape='circle')
        self.sprinkle(300, [0.05,0.01], 0, value=-1)
        background = Circle(Point(0.5,0.5), 0.5, value=1)
        self.insert(0,background)

## Microstructures - Not Implemented
class Metal(Phantom):
    def __init__(self, shape='square'):
        raise NotImplementedError

class SoftBiomaterial(Phantom):
    def __init__(self, shape='square'):
        raise NotImplementedError

class Electronics(Phantom):
    def __init__(self, shape='square'):
        raise NotImplementedError

class FiberComposite(Phantom):
    def __init__(self, shape='square'):
        raise NotImplementedError
