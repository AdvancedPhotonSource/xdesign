from xdesign.algorithms import mba
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import dxchange

input = imread('tomo_output/projections/tomo_00000.tiff')

for alpha in [0.001, 0.01, 0.1, 1., 10., 100.]:
    print 'Regularization factor is {:f}'.format(alpha)
    phase = mba(input, [1, 1], alpha=alpha)
    dxchange.write_tiff(phase, fname='pr/alpha{:f}'.format(alpha), dtype='float32')


