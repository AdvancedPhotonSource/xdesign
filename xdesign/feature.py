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

from xdesign.geometry import *
import numpy as np
import logging
from cached_property import cached_property

logger = logging.getLogger(__name__)

__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Feature']


class Feature(object):
    '''A container object for geometry objects. It is a mesh or grid object.
    Maybe it is a base class for meshes and grids?
    '''
    def __init__(self, entity, value=1):
        assert(isinstance(entity, Entity))
        # TODO: Add a base class for all entities with area?
        self.e_values = [value]
        self.entities = [entity]

    @cached_property
    def area(self):
        """Returns the total area of the feature"""
        total_area = 0
        for e in self.entities:
            total_area += e.area
        return total_area

    @cached_property
    def value(self):
        """Returns the area average value of the feature."""
        total_value = 0
        for i in range(0, len(self.entities)):
            total_value += self.e_values[i]*self.entities[i].area
        return total_value / self.area

    @cached_property
    def center(self):
        """Returns the area center of mass."""
        center = Point(0, 0)
        for i in range(0, len(self.entities)):
            center += self.entities[i].center*self.entities[i].area
        return center / self.area

    @cached_property
    def radius(self):
        """Returns the radius of the smallest boundary circle"""
        radius = 0
        for i in range(0, len(self.entities)):
            d = self.entities[i].center - self.center
            radius = max(radius, d.norm + self.entities[i].radius)
        return radius

    def translate(self, dx, dy):
        """Translate feature."""
        del self.__dict__['center']
        for e in self.entities:
            e.translate(dx, dy)

    def rotate(self, theta, point):
        """Rotate feature around a point."""
        for e in self.entities:
            e.rotate(theta, point)
