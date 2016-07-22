from phantom.plot import plot_metrics
from phantom.metrics import _compute_ssim, compute_quality, ImageQuality
from numpy.testing import assert_allclose, assert_raises, assert_equal
import numpy as np
import scipy


# SSIM metric

def test_SSIM_same_image_is_unity():
    img1 = scipy.ndimage.imread("tests/cameraman.bmp",flatten=True)
    IQ = ImageQuality(img1,img1)
    IQ = _compute_ssim(IQ)
    assert_equal(IQ.qualities[0],1,err_msg="Mean SSIM is not unity.")
    assert_equal(IQ.maps[0],np.ones(img1.shape),err_msg="local SSIMs are not unity.")
    assert_equal(img1.shape,IQ.maps[0].shape,err_msg="SSIMs map not the same size as input")

def test_compute_quality_cameraman():
    img1 = scipy.ndimage.imread("tests/cameraman.png")
    img2 = scipy.ndimage.imread("tests/cameraman_SP.png")
    img3 = scipy.ndimage.imread("tests/cameraman_H.png")
    metrics = compute_quality(img1,[img2,img3])

    plot_metrics(metrics)
