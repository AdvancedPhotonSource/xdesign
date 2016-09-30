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
import time
from glob import glob
np.set_printoptions(threshold=np.inf)


shape = [256, 256, 256]
voxel = [1, 1, 1]
energy = 20


class Probe:
    def __init__(self, energy):
        self.energy = energy
        self.wavelength = 1.23984/energy


probe = Probe(20)


protein = Material('H48.6C32.9N8.9O8.9S0.6', 1.35)
epoxy = Material('C2H4O', 1.25)

if len(glob('sav/grid_delta*')) != 0:
    print 'Reading grid...'
    t0 = time.time()
    temp = np.loadtxt('sav/dim.txt')
    shape = temp[:3]
    voxel = temp[3:6]
    energy = temp[-1]
    print('    Shape: ' + np.array2string(shape))
    print('    Voxel (nm): ' + np.array2string(voxel))
    print('    Energy (keV): ' + str(energy))
    n_files = len(glob('sav/grid_delta*'))
    grid_delta = dxchange.read_tiff_stack('sav/grid_delta_00000.tiff', range(n_files), 5)
    grid_beta = dxchange.read_tiff_stack('sav/grid_beta_00000.tiff', range(n_files), 5)
    grid = Grid3d(grid_delta.shape, voxel, energy)
    grid.grid_delta = grid_delta
    grid.grid_beta = grid_beta
    print '    Done in ' + str(time.time() - t0) + '.'
else:
    print 'Building grid...'
    t0 = time.time()
    grid = Grid3d(shape, voxel, 20)
    grid.add_cuboid([50, -78, -78], [206, 78, 78], epoxy)
    grid.add_sphere([128, 0, 0], 50, protein)
    grid.add_rod([55, 0, -25], [200, 0, -25], 10, epoxy)
    grid.add_rod([128, -50, 5], [128, 50, 5], 10, epoxy)
    print '    Done in ' + str(time.time() - t0) + '.'
    print 'Saving grid...'
    dxchange.write_tiff_stack(grid.grid_delta, 'sav/grid_delta', overwrite=True, dtype='float32')
    dxchange.write_tiff_stack(grid.grid_beta, 'sav/grid_beta', overwrite=True, dtype='float32')
    save_pack = np.asarray(shape)
    save_pack = np.append(save_pack, np.asarray(voxel))
    save_pack = np.append(save_pack, np.array([energy]))
    np.savetxt('sav/dim.txt', save_pack)

print 'Slice check.'
data_matrix = grid.grid_delta
fig = tifffile.imshow(data_matrix)
plt.show()

print 'Initializing wavefront...'
kwargs = {'type': 'plane'}
wavefront = initialize_wavefront(grid, **kwargs)

print 'Tomography simulation started.'
t0 = time.time()
ang_start = 0
ang_end = 180
n_ang = 1000
alpha = 0.001
tomography_3d(grid, wavefront, probe, ang_start, ang_end, n_ang=1000, format='h5', pr='mba', fname='tomo_alpha{:f}'.format(alpha), alpha=alpha)
print '    Done in ' + str(time.time() - t0) + '.'

# fig = tifffile.imshow(data_matrix)
# plt.show()

#dxchange.write_tiff_stack(data_matrix, overwrite=True)
