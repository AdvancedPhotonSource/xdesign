from xdesign.phantom import *
from xdesign.material import *
from numpy.testing import assert_allclose, assert_raises, assert_equal
import numpy as np
import scipy

def test_HyperbolicCocentric():
    p0 = Phantom()
    p0.load('tests/HyperbolicConcentric.txt')
    ref = p0.discrete(200, uniform=False)

    np.random.seed(0)
    p = HyperbolicConcentric()
    target = p.discrete(200, uniform=False)

    assert_equal(target, ref, "Default HyperbolicConcentric phantom has changed.")

def test_DynamicRange():
    for i in range(0,2):
        p0 = Phantom()
        p0.load('tests/DynamicRange'+str(i)+'.txt')
        ref = p0.discrete(100)

        np.random.seed(0)
        p = DynamicRange(jitter=i)
        target = p.discrete(100)

        assert_equal(target, ref, "Default DynamicRange" + str(i)+ " phantom has changed.")

def test_Soil():
    p0 = Phantom()
    p0.load('tests/Soil.txt')
    ref = p0.discrete(100)

    np.random.seed(0)
    p = Soil()
    target = p.discrete(100)

    assert_equal(target, ref, "Default Soil phantom has changed.")

def test_Foam():
    p0 = Phantom()
    p0.load('tests/Foam.txt')
    ref = p0.discrete(100)

    np.random.seed(0)
    p = Foam()
    target = p.discrete(100)

    assert_equal(target, ref, "Default Foam phantom has changed.")
