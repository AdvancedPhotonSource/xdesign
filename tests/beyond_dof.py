from xdesign.grid import Grid3d
from xdesign.material import Material
from xdesign.geometry import *
from xdesign.feature import Feature
from xdesign.phantom import Phantom


protein = Material('H48.6C32.9N8.9O8.9S0.6', 1.35)
epoxy = Material('C2H4O', 1.25)

phantom = Phantom()
tube0 = TruncatedCone_3d(top_center=Point(0, 0, 50),
                         length=400,
                         top_radius=20,
                         bottom_radius=200)
tube0 = Feature(tube0, material=epoxy)
phantom.append(tube0)

grid = Grid3d(phantom)
try:
    grid.read_grid()
except:
    grid.generate_phantom_array(size=(512, 512, 512),
                                voxel=(1, 1, 1),
                                energy=25)
    grid.save_slice_images()


# grid.save_grid()
grid.show_grid()