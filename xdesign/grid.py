from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from xdesign.geometry import *
from xdesign.geometry import Entity
from xdesign.feature import *
from xdesign.material import Material
import numpy as np
import scipy.ndimage
import logging
import warnings
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


class Grid3d(object):

    def __init__(self, size, voxel, energy):
        if not isinstance(size, np.ndarray):
            size = np.asarray(size)
        if not isinstance(voxel, np.ndarray):
            voxel = np.asarray(voxel)
        if size.size != 3 or voxel.size != 3:
            raise ValueError
        self.energy = energy
        self.grid_delta = self.grid_beta = np.zeros(size, dtype='float32')
        self.voxel_z, self.voxel_y, self.voxel_x = voxel
        y_lim, x_lim = (size[1:3] - 1) / 2 * voxel[1:3]
        z_lim = (size[0] - 1) * voxel[0]
        z_range = np.linspace(0, z_lim, size[0])
        y_range = np.linspace(-y_lim, y_lim, size[1])
        x_range = np.linspace(-x_lim, x_lim, size[2])
        self.yy, self.zz, self.xx = np.meshgrid(y_range, z_range, x_range)

    def rotate(self, theta, axis):
        raise NotImplementedError

    def translate(self, dir):
        raise NotImplementedError

    def add_sphere(self, center, radius, mat):
        assert isinstance(mat, Material)
        try:
            center_z, center_y, center_x = center
        except:
            raise ValueError
        judge = (self.zz * self.voxel_z - center_z) ** 2 + (self.yy * self.voxel_y - center_y) ** 2 + \
                (self.xx * self.voxel_x - center_x) ** 2 <= radius ** 2
        self.grid_delta[judge] = mat.refractive_index_delta(self.energy)
        self.grid_beta[judge] = mat.refractive_index_beta(self.energy)

        return self.grid_delta, self.grid_beta