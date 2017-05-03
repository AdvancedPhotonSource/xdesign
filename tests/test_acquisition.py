from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os.path

from xdesign.acquisition import raster_scan, sinogram
from xdesign.material import XDesignDefault
from numpy.testing import assert_allclose


def test_raster_scan():
    positions = [p.numpy.flatten() for p in raster_scan(2, 2)]
    positions = np.hstack(np.hstack(positions))

    correct = np.array([0.25,  -10., 0.25,       10.,
                        0.75,  -10., 0.75,       10.,
                        11.,   0.25,   -9.,     0.25,
                        11.,   0.75,   -9.,     0.75])

    assert_allclose(positions, correct)


def test_sinogram():
    p = XDesignDefault()
    sino = sinogram(32, 32, p)

    ref_file = 'tests/test_sinogram.npy'

    if not os.path.isfile(ref_file):
        ImportError('sinogram reference not found; use test_sinogram.ipynb' +
                    'to generate it')

    sino_reference = np.load(ref_file)

    assert_allclose(sino, sino_reference, atol=1e-2)
