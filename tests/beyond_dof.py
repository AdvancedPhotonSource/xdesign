import numpy as np
import matplotlib.pyplot as plt
import dxchange

from xdesign.grid import Grid3d
from xdesign.material import Material
from xdesign.geometry import *
from xdesign.feature import Feature
from xdesign.phantom import Phantom
from xdesign.propagation import *


n_particles = 100
top_z = 50.
top_radius = 20.
bottom_radius = 200.
top_thickness = 10.
bottom_thickness = 30.
length = 400.
bottom_z = top_z + length

silicon = Material('Si', 2.33)
titania = Material('TiO2', 4.23)
air = Material(delta=0, beta=0)

phantom = Phantom()
tube0 = TruncatedCone_3d(top_center=Point(0, 0, top_z),
                         length=length,
                         top_radius=top_radius,
                         bottom_radius=bottom_radius)
tube0 = Feature(tube0, material=silicon)
phantom.append(tube0)

tube1 = TruncatedCone_3d(top_center=Point(0, 0, top_z),
                         length=length,
                         top_radius=top_radius-top_thickness,
                         bottom_radius=bottom_radius-bottom_thickness)
tube1 = Feature(tube1, material=air)
phantom.append(tube1)

rand_z = np.random.choice(np.arange(top_z, bottom_z), n_particles)
for i, iz in enumerate(rand_z):
    uplim = bottom_z / iz
    rand_z[i] = iz * (np.random.rand() * (uplim - 1) + 1)

for part_z in rand_z:
    r = top_radius + (bottom_radius - top_radius) / (length - 1) * (part_z - top_z)
    theta = np.random.rand() * np.pi * 2
    part_x = np.cos(theta) * r
    part_y = np.sin(theta) * r
    rad = int(np.random.rand() * 6) + 4
    sphere = Sphere_3d(center=Point(part_x, part_y, part_z),
                       radius=rad)
    sphere = Feature(sphere, material=titania)
    phantom.append(sphere)

grid = Grid3d(phantom)
try:
    # raise Exception
    grid.generate_phantom_array(size=(512, 512, 512),
                                voxel=(1, 1, 1),
                                energy_kev=5.2,
                                skip_gen=True)
    grid.read_grid()
except:
    grid.generate_phantom_array(size=(512, 512, 512),
                                voxel=(0.01, 0.01, 0.01),
                                energy_kev=5.2)
    grid.rotate(90, axes=(1, 0))
    grid.save_slice_images()
    grid.save_grid()

wavefront = initialize_wavefront(grid, 'point_projection_lens', focal_length=50e5, lens_sample_dist=1e5)

# wavefront = initialize_wavefront(grid, 'plane', focal_length=5, lens_sample_dist=10)
dxchange.write_tiff(np.angle(wavefront), 'tmp/wavefront0_phase', dtype=np.float32, overwrite=True)
dxchange.write_tiff(np.abs(wavefront), 'tmp/wavefront0_abs', dtype=np.float32, overwrite=True)
wavefront = multislice_propagate(grid, wavefront, free_prop_dist=None)

dxchange.write_tiff(np.angle(wavefront), 'tmp/wavefront_phase', dtype=np.float32, overwrite=True)
dxchange.write_tiff(np.abs(wavefront), 'tmp/wavefront_abs', dtype=np.float32, overwrite=True)
# plt.imshow(np.abs(wavefront))
# plt.show()