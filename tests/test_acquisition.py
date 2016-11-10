from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from xdesign.acquisition import raster_scan
from numpy.testing import assert_allclose, assert_raises, assert_equal
import numpy as np


def test_raster_scan():
    positions = [p.numpy.flatten() for p in raster_scan(2, 2)]
    positions = np.hstack(np.hstack(positions))

    correct = np.array([0.25,  -10., 0.25,       10.,
                        0.75,  -10., 0.75,       10.,
                        11.,   0.25,   -9.,     0.25,
                        11.,   0.75,   -9.,     0.75])

    assert_allclose(positions, correct)
