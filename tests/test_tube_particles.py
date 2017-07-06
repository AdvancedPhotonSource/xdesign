from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import dxchange

from xdesign.material import XraylibMaterial, CustomMaterial
from xdesign.geometry import *
from xdesign.phantom import Phantom
from xdesign.propagation import *
from xdesign.plot import *
from xdesign.acquisition import Simulator


def test_model_prop_pipeline():

    n_particles = 5
    top_y = 25.e-7
    top_radius = 10.e-7
    bottom_radius = 100.e-7
    top_thickness = 5.e-7
    bottom_thickness = 15.e-7
    length = 200.e-7
    bottom_y = top_y + length

    silicon = XraylibMaterial('Si', 2.33)
    titania = XraylibMaterial('TiO2', 4.23)
    air = CustomMaterial(delta=0, beta=0)

    try:
        grid_delta = np.load('data/sav/grid/grid_delta.npy')
        grid_beta = np.load('data/sav/grid/grid_beta.npy')
    except IOError:

        tube0 = TruncatedCone_3d(top_center=Point([128.e-7, top_y, 128.e-7]),
                                 length=length,
                                 top_radius=top_radius,
                                 bottom_radius=bottom_radius)
        phantom = Phantom(geometry=tube0, material=silicon)

        tube1 = TruncatedCone_3d(top_center=Point([128.e-7, top_y, 128.e-7]),
                                 length=length,
                                 top_radius=top_radius-top_thickness,
                                 bottom_radius=bottom_radius-bottom_thickness)
        tube1 = Phantom(geometry=tube1, material=air)
        phantom.children.append(tube1)

        rand_y = []
        for i in range(n_particles):
            xi = np.random.rand()
            rand_y.append((top_radius - np.sqrt(top_radius ** 2 - top_radius ** 2 * xi + bottom_radius ** 2 * xi)) /
                          (top_radius - bottom_radius) * length + top_y)
        for part_y in rand_y:
            r = top_radius + (bottom_radius - top_radius) / (length) * (part_y - top_y)
            theta = np.random.rand() * np.pi * 2
            part_x = np.cos(theta) * r + 128.e-7
            part_z = np.sin(theta) * r + 128.e-7
            rad = int(np.random.rand() * 6.e-7) + 4.e-7
            sphere = Sphere_3d(center=Point([part_x, part_y, part_z]),
                               radius=rad)
            sphere = Phantom(geometry=sphere, material=titania)
            phantom.children.append(sphere)

        grid_delta, grid_beta = discrete_phantom(phantom, 1.e-7, bounding_box=((0, 0, 0), (255.e-7, 255.e-7, 255.e-7)),
                                                 prop=['delta', 'beta'], ratio=1, energy=25, overlay_mode='replace')

    sim = Simulator(energy=25000,
                    grid=(grid_delta, grid_beta),
                    psize=[1.e-7, 1.e-7, 1.e-7])

    sim.initialize_wavefront('plane')
    wavefront = sim.multislice_propagate()

    plt.imshow(np.abs(wavefront))
    plt.show()


if __name__ == '__main__':
    # import nose
    # nose.runmodule(exit=False)
    test_model_prop_pipeline()