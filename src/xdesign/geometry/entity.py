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
"""Define a base clase for all geometric entities."""

__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Entity']

import logging

logger = logging.getLogger(__name__)


class Entity(object):
    """Define a base class for all geometric entities.

    All geometric entities should have these attributes and methods.

    Example
    -------
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_numpy.py


    Section breaks are created with two blank lines. Section breaks are also
    implicitly created anytime a new section starts. Section bodies *may* be
    indented:

    Parameters
    ----------
    x : :class:`.ndarray`, :class:`.list`
        ND coordinates of the point.

    Notes
    -----
        This is an example of an indented section. It's like any other section,
        but the body is indented to help it stand out from surrounding text.

    If a section is indented, then a section break is created by
    resuming unindented text.


    .. note::
        There are many other directives such as versionadded, versionchanged,
        rubric, centered, ... See the sphinx documentation for more details.

    """

    def __init__(self):
        """Set the number of dimensions."""
        self._dim = 0

    def __repr__(self):
        """Return a string representation for easier debugging.

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.

        """
        raise NotImplementedError

    @property
    def dim(self):
        """Return the dimensionality of the ambient space."""
        return self._dim

    def translate(self, vector):
        """Translate along the vector.

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.
        """
        raise NotImplementedError

    def rotate(self, theta, point=None, axis=None):
        """Rotates theta radians around an axis.

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.
        """
        raise NotImplementedError

    def scale(self, vector):
        """Scale the ambient space in each dimension according to vector.

        Scaling is centered on the origin.

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.
        """
        raise NotImplementedError

    def contains(self, other):
        """Return whether the other entity is contained by this.

        Points on edges are contained by the Entity.

        Returns a boolean for all :class:`Entitity`. Returns an array of
        boolean for MxN size arrays where M is the number of points and N is
        the dimensionality.

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.
        """
        raise NotImplementedError

    def collision(self, other):
        """Return whether this collides with the other.

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.

        """
        raise NotImplementedError

    def distance(self, other):
        """Return the closest distance this and the other.

        .. note::
            This method is inherited from :class:`.Entity` which means it is
            not implemented and will throw an error.

        """
        raise NotImplementedError

    def midpoint(self, other):
        """Return the midpoint this and the other."""
        return self.distance(other) / 2.
