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

from xdesign.formats import get_NIST_table

logger = logging.getLogger(__name__)


__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['SimpleMaterial',
           'NISTMaterial']


class SimpleMaterial(object):
    """Simple material with constant mass_attenuation parameter only."""
    def __init__(self, mass_attenuation=1.0):
        super(SimpleMaterial, self).__init__()
        self._mass_attenuation = mass_attenuation
        self.density = 1.0

    def __repr__(self):
        return "SimpleMaterial(mass_attenuation={})".format(
                repr(self._mass_attenuation))

    def mass_attenuation(self, energy):
        return self._mass_attenuation


class NISTMaterial(object):
    """Materials which use NIST data to automatically calculate material
    properties based on beam energy in MeV.

    Attributes
    ----------
    density : float [g/cm^3]
        The density of the material.
    coefficent_table : Dictionary
        A Dictionary which contains the equal size arrays describing material
        properties at various beam energies [keV].

        For Example:
        coefficent_table['energy']           = array([0, 1, 2])
        coefficent_table['mass_attenuation'] = array([8, 6, 2])

    Hubbell, J.H. and Seltzer, S.M. (2004), Tables of X-Ray Mass Attenuation
    Coefficients and Mass Energy-Absorption Coefficients (version 1.4).
    [Online] Available: http://physics.nist.gov/xaamdi [2017, May 11].
    National Institute of Standards and Technology, Gaithersburg, MD.
    """

    def __init__(self, name, coefficent_table=None, density=None):
        super(NISTMaterial, self).__init__()

        self.name = name

        if coefficent_table is None or density is None:
            table, density = get_NIST_table(name)
            self.coefficent_table = table
            self.density = density

        else:
            self.coefficent_table = coefficent_table
            self.density = density

    def __repr__(self):
        return "NISTMaterial({0})".format(
                repr(self.name),
                repr(self.coefficent_table),
                repr(self.density))

    def compton_cross_section(self, energy):
        """Compton cross-section of the electron [cm^2]."""
        raise NotImplementedError

    def photoelectric_cross_section(self, energy):
        raise NotImplementedError

    def atomic_form_factor(self, energy):
        """Measure of the scattering amplitude of a wave by an isolated atom.
        Read from NIST database [Unitless]."""
        raise NotImplementedError

    def atom_concentration(self, energy):
        """Number of atoms per unit volume [1/cm^3]."""
        raise NotImplementedError

    def reduced_energy_ratio(self, energy):
        """Energy ratio of the incident x-ray and the electron energy
        [Unitless]."""
        raise NotImplementedError

    def photoelectric_absorption(self, energy):
        """X-ray attenuation due to the photoelectric effect [1/cm]."""
        raise NotImplementedError

    def compton_scattering(self, energy):
        """X-ray attenuation due to the Compton scattering [1/cm]."""
        raise NotImplementedError

    def electron_density(self, energy):
        """Electron density [e/cm^3]."""
        raise NotImplementedError

    def linear_attenuation(self, energy):
        """Total x-ray attenuation [1/cm]."""
        return self.mass_attenuation(energy) * self.density

    def refractive_index(self, energy):
        raise NotImplementedError

    def mass_attenuation(self, energy):
        """x-ray mass attenuation [cm^2/g]"""
        return self.predict_property('mass_attenuation', energy,
                                     loglogscale=True)

    def mass_ratio(self):
        raise NotImplementedError

    def number_of_elements(self):
        raise NotImplementedError

    def predict_property(self, property_name, energy, loglogscale=False):
        """Interpolate a property from the coefficient table."""
        y = self.coefficent_table[property_name]
        x = self.coefficent_table['energy']

        if loglogscale:
            return np.power(10, np.interp(np.log10(energy), np.log10(x),
                                          np.log10(y)))

        # TODO: Make special case for electron shell edges
        return np.interp(energy, x, y)
