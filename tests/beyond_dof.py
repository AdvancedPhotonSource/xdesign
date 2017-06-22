import numpy as np
import matplotlib.pyplot as plt
import dxchange

from xdesign.grid import Grid3d
from xdesign.material import XraylibMaterial, CustomMaterial
from xdesign.geometry import *
from xdesign.feature import Feature
from xdesign.phantom import Phantom
from xdesign.propagation import *
from xdesign.plot import *


n_particles = 100
top_x = 50.
top_radius = 20.
bottom_radius = 200.
top_thickness = 10.
bottom_thickness = 30.
length = 400.
bottom_x = top_x + length

silicon = XraylibMaterial('Si', 2.33)
titania = XraylibMaterial('TiO2', 4.23)
air = CustomMaterial(delta=0, beta=0)

tube0 = TruncatedCone_3d(top_center=Point([top_x, 256, 256]),
                         length=length,
                         top_radius=top_radius,
                         bottom_radius=bottom_radius)
phantom = Phantom(geometry=tube0, material=silicon)

tube1 = TruncatedCone_3d(top_center=Point([top_x, 256, 256]),
                         length=length,
                         top_radius=top_radius-top_thickness,
                         bottom_radius=bottom_radius-bottom_thickness)
tube1 = Phantom(geometry=tube1, material=air)
phantom.children.append(tube1)

rand_x = np.random.choice(np.arange(top_x, bottom_x), n_particles)
for i, ix in enumerate(rand_x):
    uplim = bottom_x / ix
    rand_x[i] = ix * (np.random.rand() * (uplim - 1) + 1)

for part_x in rand_x:
    r = top_radius + (bottom_radius - top_radius) / (length - 1) * (part_x - top_x)
    theta = np.random.rand() * np.pi * 2
    part_z = np.cos(theta) * r + 256
    part_y = np.sin(theta) * r + 256
    rad = int(np.random.rand() * 6) + 4
    sphere = Sphere_3d(center=Point([part_x, part_y, part_z]),
                       radius=rad)
    sphere = Phantom(geometry=sphere, material=titania)
    phantom.children.append(sphere)

delta, beta = discrete_phantom(phantom, 512, prop=['delta', 'beta'], ratio=1, fix_psize=True, energy=25,
                               overlay_mode='replace')

dxchange.write_tiff_stack(delta, 'tmp/temp/delta.tiff', overwrite=True, dtype='float32')



# grid = Grid3d(phantom)
# try:
#     # raise Exception
#     grid.generate_phantom_array(size=(512, 512, 512),
#                                 voxel=(1, 1, 1),
#                                 energy_kev=5.2,
#                                 skip_gen=True)
#     grid.read_grid()
# except:
#     grid.generate_phantom_array(size=(512, 512, 512),
#                                 voxel=(0.01, 0.01, 0.01),
#                                 energy_kev=5.2)
#     grid.rotate(90, axes=(1, 0))
#     grid.save_slice_images()
#     grid.save_grid()
#
# wavefront = initialize_wavefront(grid, 'point_projection_lens', focal_length=50e5, lens_sample_dist=1e5)
#
# # wavefront = initialize_wavefront(grid, 'plane', focal_length=5, lens_sample_dist=10)
# dxchange.write_tiff(np.angle(wavefront), 'tmp/wavefront0_phase', dtype=np.float32, overwrite=True)
# dxchange.write_tiff(np.abs(wavefront), 'tmp/wavefront0_abs', dtype=np.float32, overwrite=True)
# wavefront = multislice_propagate(grid, wavefront, free_prop_dist=None)
#
# dxchange.write_tiff(np.angle(wavefront), 'tmp/wavefront_phase', dtype=np.float32, overwrite=True)
# dxchange.write_tiff(np.abs(wavefront), 'tmp/wavefront_abs', dtype=np.float32, overwrite=True)
# # plt.imshow(np.abs(wavefront))
# # plt.show()