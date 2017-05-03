from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from xdesign.acquisition import raster_scan, sinogram
from xdesign.phantom import Phantom
from xdesign.geometry import Circle, Triangle, Point
from xdesign.feature import Feature
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


def test_sinogram():
    # load first because test is pointless if missing reference.
    sino_reference = np.load('tests/test_sinogram.npy')

    circle = Feature(Circle(Point([0.7, 0.5]), radius=0.1))
    triangle = Feature(Triangle(Point([0.2, 0.4]),
                                Point([0.2, 0.6]),
                                Point([0.4, 0.6])))
    circtri = Phantom()
    circtri.append(circle)
    circtri.append(triangle)
    sino = sinogram(32, 32, circtri)

    assert_allclose(sino, sino_reference, atol=1e-2)
