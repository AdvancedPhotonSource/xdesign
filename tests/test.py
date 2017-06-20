import skimage.morphology as morph
import xraylib
import numpy as np


class Probe:
    def __init__(self, energy):
        self.energy = energy
        self.wavelength = 1.23984/energy

test_delta = 1 - xraylib.Refractive_Index_Re('H2O', 10, 0.92)
test_beta = xraylib.Refractive_Index_Im('H2O', 10, 0.92)

delta_grid = beta_grid = np.zeros([40, 40])
delta_grid[10:31, 10:31] = beta_grid[10:31, 10:31] = morph.disk(10, dtype=np.float32)
delta_grid[np.where(delta_grid != 0)] = test_delta
beta_grid[np.where(delta_grid != 0)] = test_beta

delta_grid[:, :] = 0.
beta_grid[:, :] = 0.

print delta_grid

probe = Probe(10)

kwargs = {'type':'point', 'width':4}
wavefront = propagation.initialize_wavefront(40, **kwargs)

wavefront = propagation.multislice_propagate(delta_grid, beta_grid, probe, 5, 1, wavefront)

propagation.plot_wavefront(wavefront, 1, '/Users/Ming/Research/Data', 'out')
