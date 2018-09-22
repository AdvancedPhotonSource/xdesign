from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import numpy as np
import matplotlib.pyplot as plt
import xdesign as xd
import logging

SIZE = 32


def test_sinogram():
    """Compare sinogram of XDesignDefault with a reference."""
    # Create the phantom
    phan = xd.XDesignDefault()
    # Plot it for debugging purposes
    plt.figure()
    xd.sidebyside(phan, SIZE)
    # Generate the scanning trajectory
    theta, h = xd.raster_scan2D(SIZE, SIZE)
    # Create a probe
    prb = xd.Probe(size=1/SIZE)
    # Meausure the phantom with the probe
    sino = prb.measure(phan, theta, h)
    sino = -np.log(sino)
    # Plot the sinogram for debugging
    plt.figure()
    plt.imshow(sino, origin='lower')
    # Load the reference from file
    ref_file = 'tests/test_sinogram.npy'
    if not os.path.isfile(ref_file):
        ImportError('sinogram reference not found; use test_sinogram.ipynb' +
                    'to generate it')
    sino_reference = np.load(ref_file)
    # assert that they are equal
    np.testing.assert_allclose(sino, sino_reference)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    test_sinogram()
    plt.show()
