from xdesign.grid import *
from xdesign.material import Material
from xdesign.propagation import *
from xdesign.acquisition import tomography_3d
import numpy as np
import matplotlib.pyplot as plt
from xdesign.plot import *
import tifffile
from scipy.ndimage.interpolation import rotate
import dxchange
np.set_printoptions(threshold=np.inf)


class Probe:
    def __init__(self, energy):
        self.energy = energy
        self.wavelength = 1.23984/energy


probe = Probe(20)


protein = Material('H48.6C32.9N8.9O8.9S0.6', 1.35)
epoxy = Material('C2H4O', 1.25)

grid = Grid3d([128, 128, 128], [1, 1, 1], 20)
grid.add_cuboid([32, -32, -32], [96, 32, 32], epoxy)
grid.add_sphere([64, 0, 0], 16, protein)
grid.add_rod([35, 0, -10], [92, 0, -10], 3, epoxy)
grid.add_rod([64, -29, 5], [64, 29, 5], 3, epoxy)

data_matrix = grid.grid_delta

# kwargs = {'type': 'plane'}
# wavefront = initialize_wavefront(grid, **kwargs)
# wavefront = multislice_propagate(grid, probe, wavefront)
# plot_wavefront(wavefront, grid)

# ang_start = 0
# ang_end = 180
# ang_step = 5
# tomography_3d(grid, ang_start, ang_end, ang_step)

fig = tifffile.imshow(data_matrix)
plt.show()

#dxchange.write_tiff_stack(data_matrix, overwrite=True)
