import numpy as np


def gen_mesh(max, shape):
    """Generate mesh grid.

    Parameters:
    -----------
    lengths : ndarray
        Half-lengths of axes in nm or nm^-1.
    shape : ndarray
        Number of pixels in each dimension.
    """
    dim = len(max)
    axes = [np.array([np.linspace(-max[0], max[0], shape[0])])]
    for i in range(1, dim):
        axes.append(np.linspace(-max[i], max[i], shape[i]))
    res = np.meshgrid(*axes)
    return res