from xdesign.plot import plot_metrics
from xdesign.metrics import _compute_ssim, _compute_vifp, _compute_fsim, compute_quality, ImageQuality
from numpy.testing import *
import numpy as np
import scipy


# SSIM metric

def test_SSIM_same_image_is_unity():
    img1 = scipy.ndimage.imread("tests/cameraman.png")
    IQ = ImageQuality(img1,img1)
    IQ = _compute_ssim(IQ)
    assert_equal(IQ.qualities[0],1,err_msg="Mean is not unity.")
    assert_equal(IQ.maps[0],np.ones(img1.shape),err_msg="local metrics are not unity.")
    assert_equal(img1.shape,IQ.maps[0].shape,err_msg="SSIMs map not the same size as input")

def test_VIFp_same_image_is_unity():
    img1 = scipy.ndimage.imread("tests/cameraman.png")
    IQ = ImageQuality(img1,img1)
    IQ = _compute_vifp(IQ)
    assert_almost_equal(IQ.qualities,1,err_msg="Mean is not unity.")
    #assert_equal(IQ.maps,1,err_msg="local metrics are not unity.")

def test_FSIM_same_image_is_unity():
    img1 = scipy.ndimage.imread("tests/cameraman.png")
    IQ = ImageQuality(img1,img1)
    IQ = _compute_fsim(IQ)
    assert_almost_equal(IQ.qualities,1.,err_msg="Mean is not unity.")
    #assert_almost_equal(IQ.maps,np.ones(len(IQ.maps)),err_msg="local metrics are not unity.")

def test_compute_quality_cameraman():
    img1 = scipy.ndimage.imread("tests/cameraman.png") # original
    img4 = scipy.ndimage.imread("tests/cameraman_mixed1.png")
    # salt and pepper, gaussian noise, square? smoothing filter, guassian filter
    metrics = compute_quality(img1,[img4],method="VIFp", L=256)
    #plot_metrics(metrics)
    metrics = compute_quality(img1,[img4],method="FSIM", L=256)
    #plot_metrics(metrics)
    metrics = compute_quality(img1,[img4],method="MSSSIM", L=256)
    #plot_metrics(metrics)
