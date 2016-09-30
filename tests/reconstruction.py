import tomopy
import dxchange
import numpy as np

filename = 'tomo_output/tomo.h5'
savefolder = 'tomo_output/recon/'
slice = 128
n_ang = 1000

sino, _, _ = dxchange.read_aps_32id(filename, sino=(slice, slice + 1))

ang = tomopy.angles(n_ang)

rec = tomopy.recon(sino, ang, center = 127, algorithm='gridrec')

rec = np.squeeze(rec)

dxchange.write_tiff(rec, fname='{:s}recon_slice_{:05d}.tiff'.format(savefolder, slice))
