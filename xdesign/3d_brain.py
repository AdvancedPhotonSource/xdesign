import xdesign
from xdesign.grid import *
from xdesign.material import Material
import numpy as np
import matplotlib.pyplot as plt
from xdesign.plot import *

water = xdesign.material.Material('H2O', 1)

grid = Grid3d([64, 64, 64], [1, 1, 1], 20)

delta_grid, beta_grid = grid.add_sphere([32, 0, 0], 16, water)

fig = plt.figure()
plt.contour(delta_grid[32, :, :])
plt.show()
