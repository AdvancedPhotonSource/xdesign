#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2019, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2019. UChicago Argonne, LLC. This software was produced       #
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

"""Generate codes for space- and time-coded apertures.

.. moduleauthor:: Daniel Ching
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2019, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['mura_1d', 'mura_2d', 'raskar']

def is_prime(n):
    """Return True if n is prime."""
    if n == 2 or n == 3:
        return True
    if n < 2 or n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    r = int(n**0.5)
    f = 5
    while f <= r:
        if n % f == 0:
            return False
        if n % (f+2) == 0:
            return False
        f += 6
    return True


def quadratic_residues_modulo(n):
    """Return all quadratic residues modulo n in the range 0, ..., n-1.

    q is a quadratic residue modulo n if it is congruent to a perfect square
    modulo n.
    """
    x = np.arange(n)
    q = x**2 % n
    return q


def mura_1d(L):
    """Return the longest MURA whose length is less than or equal to L.

    From Wikipedia:
    A Modified uniformly redundant array (MURA) can be generated in any length
    L that is prime and of the form::

        L = 4m + 1, m = 1, 2, 3, ...,

    the first six such values being ``L = 5, 13, 17, 29, 37``. The binary sequence
    of a linear MURA is given by ``A[0:L]`` where::

        A[i] = {
            0 if i = 0,
            1 if i is a quadratic residue modulo L, i != 0,
            0 otherwise,
        }
    """
    if L < 5:
        raise ValueError("A MURA cannot have length less than 5.")
    # overestimate m to guess a MURA longer than L
    m = (L + 1) // 4
    L1 = (4 * m) + 1
    # find an allowed MURA length, L1, <= L
    while not (L1 <= L and is_prime(L1)):
        m = m - 1
        L1 = (4 * m) + 1
    # Compute the MURA
    A = np.zeros(L1, dtype=np.bool)
    A[quadratic_residues_modulo(L1)] = 1
    A[0] = 0
    print("MURA is length {}".format(L1))
    assert L1 <= L, "len(MURA) should be <= {}, but it's {}.".format(L, L1)
    return A


def mura_2d(M, N=None):
    """Return the largest 2D MURA whose lengths are less than M and N.

    From Wikipedia:
    A rectangular MURA, ``A[0:M, 0:N]``, is defined as follows::

        A[i, j] = {
            0 if i = 0,
            1 if j = 0, i != 0,
            1 if C[i] * C[j] = 1,
            0 othewise,
        }

        C[i] = {
            1 if i is a quadratic residue modulo p,
            -1 otherwise,
        }

    where p is the length of the matching side M, N.
    """
    # Use 1D Muras to start
    Ci = mura_1d(M).astype(np.int8)
    M1 = len(Ci)
    if N is None:
        N1 = M1
        Cj = np.copy(Ci)
    else:
        Cj = mura_1d(N).astype(np.int8)
        N1 = len(Cj)
    # Modify 1D Muras to match 2D mura coefficients; ignore i, j = 0 those are
    # set later.
    Ci[Ci != 1] = -1
    Cj[Cj != 1] = -1
    # Arrays must be 2D for matrix multiplication
    Ci = Ci[..., np.newaxis]
    Cj = Cj[np.newaxis, ...]
    A = (Ci @ Cj) == 1
    assert A.shape[0] == M1 and A.shape[1] == N1, \
        "A is not the correct shape! {} != ({}, {})".format(A.shape, M1, N1)
    A[0, :] = 0
    A[:, 0] = 1
    return A


def raskar(npool):
    """Return the coded mask from Raskar et al."""
    return np.array([1, 0, 1, 0, 0,  0, 1, 0, 1, 1,  # 10
                     0, 0, 0, 0, 0,  1, 0, 1, 0, 0,  # 20
                     0, 0, 1, 1, 0,  0, 1, 1, 1, 1,  # 30
                     0, 1, 1, 1, 0,  1, 0, 1, 1, 1,  # 40
                     0, 0, 1, 0, 0,  1, 1, 0, 0, 1,  # 50
                     1, 1], dtype='bool')  # must be boolean
