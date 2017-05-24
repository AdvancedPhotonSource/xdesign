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


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
import warnings

try:
    import xraylib as xl
except ImportError:
    warnings.warn("xraylib is requried for XraylibMaterial", ImportWarning)

from xdesign.formats import get_NIST_table


logger = logging.getLogger(__name__)


__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['SimpleMaterial',
           'NISTMaterial',
           'XraylibMaterial']


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
                repr(self._mass_attenuation))

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
        return "XraylibMaterial({0}, {1})".format(repr(self.compound),
                                                  repr(self.density))

    @memodict
    def beta(self, energy):
        """Absorption coefficient."""
        return xl.Refractive_Index_Im(self.compound, energy, self.density)

    @memodict
    def delta(self, energy):
        """Decrement of refractive index."""
        return 1 - xl.Refractive_Index_Re(self.compound, energy, self.density)


class NISTMaterial(Material):
    """Materials which use NIST data to automatically calculate material
    properties based on beam energy in KeV.

    If no density is provided, then density defaults to the density in the NIST
    database.

    Attributes
    ----------
    name : string
        The NIST string decribing the material.
    density : float [g/cm^3] (default: None)
        The mass density of the material.
    nist_density : float [g/cm^3] (default: None)
        The mass density of the material accordint to NIST.
    coefficent_table : :py:class:`Dictionary`
        A Dictionary which contains the equal size arrays describing material
        properties at various beam energies [keV].

        For Example:
        coefficent_table['energy']           = array([0, 1, 2])
        coefficent_table['mass_attenuation'] = array([8, 6, 2])

    References
    ----------
    Hubbell, J.H. and Seltzer, S.M. (2004), Tables of X-Ray Mass Attenuation
    Coefficients and Mass Energy-Absorption Coefficients (version 1.4).
    [Online] Available: http://physics.nist.gov/xaamdi [2017, May 11].
    National Institute of Standards and Technology, Gaithersburg, MD.
    """

    def __init__(self, name, density=None):
        super(NISTMaterial, self).__init__()

        self.name = name

        table, nist_density = get_NIST_table(name)
        self.coefficent_table = table
        self.nist_density = nist_density

        if density is None:
            self.density = nist_density
        else:
            self.density = density

    def __repr__(self):
        return "NISTMaterial({0}, density={1})".format(repr(self.name),
                                                       repr(self.density))

    @memodict
    def linear_attenuation(self, energy):
        """linear x-ray attenuation [1/cm] for the energy [KeV]."""
        return self.mass_attenuation(energy) * self.density

    @memodict
    def mass_attenuation(self, energy):
        """mass x-ray attenuation [1/cm] for the energy [KeV]."""
        return self.predict_property('mass_attenuation', energy,
                                     loglogscale=True)

    def predict_property(self, property_name, energy, loglogscale=False):
        """Interpolate a property from the coefficient table."""
        y = self.coefficent_table[property_name]
        x = self.coefficent_table['energy']

        if loglogscale:
            return np.power(10, np.interp(np.log10(energy), np.log10(x),
                                          np.log10(y)))

        # TODO: Make special case for electron shell edges
        return np.interp(energy, x, y)
