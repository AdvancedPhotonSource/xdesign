import numpy as np

from xdesign.grid import Grid3d
from xdesign.material import Material
from xdesign.geometry import *
from xdesign.feature import Feature
from xdesign.phantom import Phantom


n_particles = 20.
top_z = 50.
top_radius = 20.
bottom_radius = 200.
top_thickness = 10.
bottom_thickness = 30.
length = 400.

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

for part_z in np.random.choice(range(50, 450), n_particles):
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
    grid.read_grid()
except:
    grid.generate_phantom_array(size=(512, 512, 512),
                                voxel=(1, 1, 1),
                                energy_kev=5.2)
    grid.save_slice_images()
    grid.save_grid()
