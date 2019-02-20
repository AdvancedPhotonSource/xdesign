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

from pkg_resources import resource_filename, resource_exists
from codecs import open
import json
import requests
import logging


logger = logging.getLogger(__name__)


__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['get_NIST_table']

"""Placeholder module for reading/writing various formatted phantoms,
experiments, meshes, data, etc."""


def get_NIST_table(class_name):
    """Return a dictionary with the NIST data and the density.

    Energy values are converted to keV. Attenuation values remain cm^2/g.
    """

    NIST_folder = resource_filename("xdesign", "NIST")

    with open(NIST_folder + '/NIST_index.json', 'r', encoding="utf-8") as f:
        index = json.load(f)

    try:
        density = index[class_name]['density']
    except KeyError:
        raise ValueError('{} is not in the NIST index. '.format(class_name) +
                         'Check NIST_index.json for spelling errors.')

    NIST_file = "/{}.json".format(class_name)

    if not resource_exists("xdesign", "NIST" + NIST_file):
        logger.info('Grabbing %s NIST data from the internet.', class_name)
        # Determine which URL to use.

        url = "http://xrayplots.2mrd.com.au/api/"

        if not index[class_name]['symbol']:
            url += "material/" + class_name
        else:
            url += "element/{Z}".format(Z=index[class_name]['z'])

        # Fetch the NIST data from the internet.
        response = requests.get(url)
        jsondata = response.json()

        # Reformat the JSON.
        table = dict()
        table['energy'] = [point[u'e'] * 1000 for point in jsondata]
        table['mass_attenuation'] = [point[u'a'] for point in jsondata]
        table['mass_energy_absorption'] = [point[u'm'] for point in jsondata]

        # Save the JSON for later.
        with open(NIST_folder + NIST_file, 'w', encoding="utf-8") as f:
            json.dump(table, f)

    else:
        logger.info('Found %s NIST data locally.', class_name)

        with open(NIST_folder + NIST_file, 'r', encoding="utf-8") as f:
            # If it's already downloaded, then just load it.
            table = json.load(f)

    return table, density
