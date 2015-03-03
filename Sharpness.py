__author__ = 'Helder C. R. de Oliveira'
__email__ = 'heldercro@gmail.com'
__url__ = 'http://helderc.github.io'
__date__ = '2015-mar-03'

'''
    Code based on paper:

    A Nonlocal Maximum Likelihood Estimation Method for Rician Noise Reduction
    in MR Images. IEEE Trans. on Med. Imaging, VOL. 28, NO. 2, feb 2009.
    by Lili He and Ian R. Greenshields
'''

import numpy as np
from scipy import ndimage


def sharpness(img):
    isinstance(img, np.ndarray)

    [gy, gx] = np.gradient(img)

    hx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    hy = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])

    wx = ndimage.convolve(img, hx)
    wy = ndimage.convolve(img, hy)

    sharp = ((wx ** 2) * (gx ** 2)) + ((wy ** 2) * (gy ** 2))

    # At that paper the Sharpness is calculated as 'sharp' is. However, it is a
    # huge number. Appliyng  log as above, will reduce that number and then
    # the sharpness will be given in dB.
    sharp_db = 10 * np.log10(sharp.sum())

    return sharp_db