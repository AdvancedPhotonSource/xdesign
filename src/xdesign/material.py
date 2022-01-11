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
"""Defines objects which auto-generate a parameterized :class:`.Phantom`.

.. moduleauthor:: Daniel J Ching <carterbox@users.noreply.github.com>
.. moduleauthor:: Doga Gursoy <dgursoy@aps.anl.gov>
"""

__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'SimpleMaterial',
    'XraylibMaterial',
]

import logging
import warnings

import numpy as np

try:
    import xraylib as xl
except ImportError:
    warnings.warn("xraylib is requried for XraylibMaterial", ImportWarning)

logger = logging.getLogger(__name__)


def memodict(f):
    """Memoization decorator for a function taking a single argument
    http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/
    """

    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret

    return memodict().__getitem__


class Material(object):
    """A base class for Materials.

    Attributes
    ----------
    density : float [g/cm^3] (default: 1.0)
        The mass density of the material
    """

    def __init__(self, density=1.0):
        super(Material, self).__init__()
        self.density = density


class SimpleMaterial(Material):
    """Simple material with constant mass_attenuation parameter only.

    Attributes
    ----------
    density : float [g/cm^3] (default: 1.0)
        The mass density of the material
    """

    def __init__(self, mass_attenuation=1.0):
        super(SimpleMaterial, self).__init__(density=1.0)
        self._mass_attenuation = mass_attenuation

    def __repr__(self):
        return "SimpleMaterial(mass_attenuation={})".format(
            repr(self._mass_attenuation)
        )

    def linear_attenuation(self, energy):
        """linear x-ray attenuation [1/cm] for the energy [KeV]."""
        return self._mass_attenuation

    def mass_attenuation(self, energy):
        """mass x-ray attenuation [1/cm] for the energy [KeV]."""
        return self._mass_attenuation


class XraylibMaterial(Material):
    """Materials which use `xraylib` data to automatically calculate material
    properties based on beam energy in KeV.

    Attributes
    ----------
    compound : string
        Molecular formula of the material.
    density : float [g/cm^3] (default: 1.0)
        The mass density of the material
    """

    def __init__(self, compound, density):
        self.compound = compound
        self.density = density

    def __repr__(self):
        return "XraylibMaterial({0}, {1})".format(
            repr(self.compound), repr(self.density)
        )

    @memodict
    def beta(self, energy):
        """Absorption coefficient."""
        return xl.Refractive_Index_Im(self.compound, energy, self.density)

    @memodict
    def delta(self, energy):
        """Decrement of refractive index."""
        return 1 - xl.Refractive_Index_Re(self.compound, energy, self.density)
