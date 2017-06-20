from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from xdesign.feature import *
from xdesign.material import Material
from xdesign.phantom import Phantom
import numpy as np
import numpy.linalg as nl
import scipy.ndimage
import logging
import warnings
import matplotlib.pyplot as plt
import tifffile
import dxchange
#np.set_printoptions(threshold=np.inf)
from scipy.ndimage.interpolation import rotate, shift


class Grid3d(object):

    def __init__(self, phantom):

        assert isinstance(phantom, Phantom)
        self.phantom = phantom
        self.grid_delta = None
        self.grid_beta = None
        self.energy_kev = None
        self.lmbda_nm = None
        self.voxel_z, self.voxel_y, self.voxel_x = (None, None, None)
        self.xx, self.yy, self.zz = (None, None, None)
        self.x_range, self.y_range, self.z_range = (None, None, None)

    def generate_phantom_array(self, size, voxel, energy_kev, **kwargs):

        self.size = np.array(size)
        if self.grid_delta is None:
            self.grid_delta = np.zeros(size, dtype='float')
        if self.grid_beta is None:
            self.grid_beta = np.zeros(size, dtype='float')
        self.energy_kev = energy_kev
        self.lmbda_nm = 1.24 / self.energy_kev
        self.voxel_x, self.voxel_y, self.voxel_z = voxel
        x_lim, y_lim= (self.size[0:2] - 1) / 2 * np.array(voxel[0:2])
        z_lim = (self.size[2] - 1) * voxel[2]
        self.z_range = np.linspace(0, z_lim, size[2])
        self.y_range = np.linspace(-y_lim, y_lim, size[1])
        self.x_range = np.linspace(-x_lim, x_lim, size[0])
        self.yy, self.zz, self.xx = np.meshgrid(self.y_range, self.z_range, self.x_range)

        if ('skip_gen' not in kwargs) or not(kwargs['skip_gen']):
            for feat in self.phantom.feature:
                feat.generate(self)

    def rotate(self, theta, axes=(0, 2)):
        self.grid_delta = rotate(self.grid_delta, theta, axes=axes, reshape=False)
        self.grid_beta = rotate(self.grid_beta, theta, axes=axes, reshape=False)
        return self

    def translate(self, dir):
        raise NotImplementedError

    def save_grid(self, save_path='data/sav/grid'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'grid_delta'), self.grid_delta)
        np.save(os.path.join(save_path, 'grid_beta'), self.grid_beta)
        grid_pars = [self.size, (self.voxel_x, self.voxel_y, self.voxel_z), self.energy_kev]
        np.save(os.path.join(save_path, 'grid_pars'), grid_pars)

    def read_grid(self, save_path='data/sav/grid'):
        try:
            self.grid_delta = np.load(os.path.join(save_path, 'grid_delta.npy'))
            self.grid_beta = np.load(os.path.join(save_path, 'grid_beta.npy'))
        except:
            raise ValueError('Failed to read grid.')

    def read_parameters(self, save_path='data/sav/grid'):
        grid_pars = np.load(os.path.join(save_path, 'grid_pars.npy'))
        size, voxel, energy_kev = grid_pars
        self.generate_phantom_array(size, voxel, energy_kev, skip_gen=True)

    def save_slice_images(self, save_path='data/sav/slices'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        dxchange.write_tiff_stack(self.grid_delta, os.path.join(save_path, 'delta'),
                                  overwrite=True, dtype=np.float32)
        dxchange.write_tiff_stack(self.grid_beta, os.path.join(save_path, 'beta'),
                                  overwrite=True, dtype=np.float32)

    def show_grid(self, part='delta'):
        if part == 'delta':
            tifffile.imshow(self.grid_delta)
        elif part == 'beta':
            tifffile.imshow(self.grid_beta)
        else:
            print('WARNING: wrong part specified for show_grid.')