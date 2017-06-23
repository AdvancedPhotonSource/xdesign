import numpy as np
import matplotlib.pyplot as plt
import dxchange

from xdesign.material import XraylibMaterial, CustomMaterial
from xdesign.geometry import *
from xdesign.phantom import Phantom
from xdesign.propagation import *
from xdesign.plot import *
from xdesign.acquisition import Simulator


n_particles = 100
top_y = 50.
top_radius = 20.
bottom_radius = 200.
top_thickness = 10.
bottom_thickness = 30.
length = 400.
bottom_y = top_y + length

silicon = XraylibMaterial('Si', 2.33)
titania = XraylibMaterial('TiO2', 4.23)
air = CustomMaterial(delta=0, beta=0)

try:
    grid_delta = np.load('data/sav/grid/grid_delta.npy')
    grid_beta = np.load('data/sav/grid/grid_beta.npy')

except:

    tube0 = TruncatedCone_3d(top_center=Point([256, top_y, 256]),
                             length=length,
                             top_radius=top_radius,
                             bottom_radius=bottom_radius)
    phantom = Phantom(geometry=tube0, material=silicon)

    tube1 = TruncatedCone_3d(top_center=Point([256, top_y, 256]),
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
        r = top_radius + (bottom_radius - top_radius) / (length - 1) * (part_y - top_y)
        theta = np.random.rand() * np.pi * 2
        part_x = np.cos(theta) * r + 256
        part_z = np.sin(theta) * r + 256
        rad = int(np.random.rand() * 6) + 4
        sphere = Sphere_3d(center=Point([part_x, part_y, part_z]),
                           radius=rad)
        sphere = Phantom(geometry=sphere, material=titania)
        phantom.children.append(sphere)

    grid_delta, grid_beta = discrete_phantom(phantom, 512, prop=['delta', 'beta'], ratio=1, fix_psize=True, energy=25,
                                   overlay_mode='replace')

    np.save('data/sav/grid/grid_delta.npy', grid_delta)
    np.save('data/sav/grid/grid_beta.npy', grid_beta)

sim = Simulator(energy=25,
                grid=(grid_delta, grid_beta),
                psize=[1, 1, 1])

sim.initialize_wavefront('plane')
wavefront = sim.multislice_propagate()


plt.imshow(np.abs(wavefront))
plt.show()


