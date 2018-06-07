from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
import os.path
from numpy.testing import assert_allclose

from xdesign.acquisition import *
from xdesign.phantom import *
from xdesign.plot import *
import matplotlib.pyplot as plt

SIZE = 64


def test_sinogram():
    theta, h, v = raster_scan2D(SIZE, SIZE)
    prb = Probe(size=1/SIZE)
    phan = XDesignDefault()
    phan.translate([-.5, -.5])
    sidebyside(phan, SIZE)

    sino = prb.measure(phan, theta, h, v)
    sino = -np.log(sino)

    plt.figure()
    plt.imshow(sino.reshape(SIZE, SIZE), origin='lower')

    # ref_file = 'tests/test_sinogram.npy'
    # if not os.path.isfile(ref_file):
    #     ImportError('sinogram reference not found; use test_sinogram.ipynb' +
    #                 'to generate it')
    #
    # sino_reference = np.load(ref_file)
    #
    # assert_allclose(sino, sino_reference, atol=1e-2)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    test_sinogram()
    plt.show()
