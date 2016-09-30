from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from xdesign.geometry import *
from xdesign.geometry import Entity
from xdesign.feature import *
from xdesign.material import Material
import numpy as np
import numpy.linalg as nl
import scipy.ndimage
import logging
import warnings
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=np.inf)
from scipy.ndimage.interpolation import rotate, shift


class Grid3d(object):

    def __init__(self, size, voxel, energy):
        if not isinstance(size, np.ndarray):
            size = np.asarray(size)
        if not isinstance(voxel, np.ndarray):
            voxel = np.asarray(voxel)
        if size.size != 3 or voxel.size != 3:
            raise ValueError
        self.size = size
        self.energy = energy
        self.grid_delta = np.zeros(size, dtype='float32')
        self.grid_beta = np.zeros(size, dtype='float32')
        self.voxel_z, self.voxel_y, self.voxel_x = voxel
        y_lim, x_lim = (size[1:3] - 1) / 2 * voxel[1:3]
        z_lim = (size[0] - 1) * voxel[0]
        self.z_range = np.linspace(0, z_lim, size[0])
        self.y_range = np.linspace(-y_lim, y_lim, size[1])
        self.x_range = np.linspace(-x_lim, x_lim, size[2])
        self.yy, self.zz, self.xx = np.meshgrid(self.y_range, self.z_range, self.x_range)

    def rotate(self, theta):
        self.grid_delta = rotate(self.grid_delta, theta, axes=(0, 2), reshape=False)
        self.grid_beta = rotate(self.grid_beta, theta, axes=(0, 2), reshape=False)
        return self

    def translate(self, dir):
        raise NotImplementedError

    def add_sphere(self, center, radius, mat):
        assert isinstance(mat, Material)
        try:
            center_z, center_y, center_x = center
        except:
            raise ValueError
        judge = (self.zz - center_z) ** 2 + (self.yy - center_y) ** 2 + \
                (self.xx - center_x) ** 2 <= radius ** 2
        self.grid_delta[judge] = mat.refractive_index_delta(self.energy)
        self.grid_beta[judge] = mat.refractive_index_beta(self.energy)
        print('Sphere added.')
        return self

    def add_rod(self, x1, x2, radius, mat):
        assert isinstance(mat, Material)
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        x0 = np.empty(np.append(self.xx.shape, 3))
        x0[:, :, :, 0] = self.zz
        x0[:, :, :, 1] = self.yy
        x0[:, :, :, 2] = self.xx
        judge = np.cross(x2 - x1, x1 - x0)
        judge = judge.reshape(int(judge.size / 3), 3)
        judge = np.asarray(map(nl.norm, judge))
        judge = judge.reshape(self.xx.shape)
        judge = judge / nl.norm(x2 - x1) <= radius
        judge_seg = -(x1 - x0).dot(x2 - x1) / nl.norm(x2 - x1) ** 2
        judge = judge * (judge_seg >= 0) * (judge_seg <= 1)
        self.grid_delta[judge] = mat.refractive_index_delta(self.energy)
        self.grid_beta[judge] = mat.refractive_index_beta(self.energy)
        print('Rod added.')
        return self

    def add_cuboid(self, x1, x2, mat):
        assert isinstance(mat, Material)
        judge = (self.zz >= x1[0]) * (self.zz <= x2[0]) * (self.yy >= x1[1]) * (self.yy <= x2[1]) * \
                (self.xx >= x1[2]) * (self.xx <= x2[2])
        self.grid_delta[judge] = mat.refractive_index_delta(self.energy)
        self.grid_beta[judge] = mat.refractive_index_beta(self.energy)
        print('Cuboid added.')
        return self
