import matplotlib.pyplot as plt
import os.path
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import xdesign as xd
from xdesign.geometry import *
from xdesign.acquisition import beamcirc, beampoly, Probe


# Nonintersecting beams
def test_beamcirc_nonintersecting_top():
    circ = Circle(Point([0, 3]), 1)
    beam = Probe(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 0., rtol=1e-6)


def test_beamcirc_nonintersecting_bottom():
    circ = Circle(Point([0, -3]), 1)
    beam = Probe(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 0., rtol=1e-6)


def test_beampoly_nonintersecting_top():
    tri = Triangle(Point([0, 1]), Point([1, -1]), Point([-1, -1]))
    beam = Probe(Point([-2, 2]), Point([2, 2]), 2)
    assert_allclose(beampoly(beam, tri), 0., rtol=1e-6)


def test_beampoly_nonintersecting_bottom():
    tri = Triangle(Point([0, 1]), Point([1, -1]), Point([-1, -1]))
    beam = Probe(Point([-2, -2]), Point([2, -2]), 2)
    assert_allclose(beampoly(beam, tri), 0., rtol=1e-6)


# Partial intersections
def test_beamcirc_intersecting_partially_from_top_outside_center():
    circ = Circle(Point([0, 1.5]), 1)
    beam = Probe(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 0.614184849304, rtol=1e-6)


def test_beamcirc_intersecting_partially_from_bottom_outside_center():
    circ = Circle(Point([0, -1.5]), 1)
    beam = Probe(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 0.614184849304, rtol=1e-6)


def test_beamcirc_intersecting_partially_from_top_inside_center():
    circ = Circle(Point([0, 0.5]), 1)
    beam = Probe(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 2.52740780429, rtol=1e-6)


def test_beamcirc_intersecting_partially_from_bottom_inside_center():
    circ = Circle(Point([0, -0.5]), 1)
    beam = Probe(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 2.52740780429, rtol=1e-6)


def test_beampoly_intersecting_partially_from_top():
    tri = Square(Point([0.5, 0.5]), side_length=1.0)
    beam = Probe(Point([-2, 1]), Point([2, 1]), 1)
    assert_allclose(beampoly(beam, tri), 1/2, rtol=1e-6)


def test_beampoly_intersecting_partially_from_bottom():
    tri = Square(Point([0.5, 0.5]), side_length=1.0)
    beam = Probe(Point([-2, 0]), Point([2, 0]), 1)
    assert_allclose(beampoly(beam, tri), 1/2, rtol=1e-6)


# Full intersections
def test_beamcirc_intersecting_fully_from_top_outside_center():
    circ = Circle(Point([0, 1.5]), 3)
    beam = Probe(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 10.0257253792, rtol=1e-6)


def test_beamcirc_intersecting_fully_from_bottom_outside_center():
    circ = Circle(Point([0, -1.5]), 3)
    beam = Probe(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 10.0257253792, rtol=1e-6)


def test_beamcirc_intersecting_fully_from_top_inside_center():
    circ = Circle(Point([0, 0.5]), 3)
    beam = Probe(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 11.5955559562, rtol=1e-6)


def test_beamcirc_intersecting_fully_from_bottom_inside_center():
    circ = Circle(Point([0, -0.5]), 3)
    beam = Probe(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 11.5955559562, rtol=1e-6)


def test_beamcirc_intersecting_fully():
    circ = Circle(Point([0, 0]), 1)
    beam = Probe(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 3.14159265359, rtol=1e-6)


def test_beampoly_intersecting_fully():
    tri = Square(Point([0, 0]), side_length=2.0)
    beam = Probe(Point([-2, 0]), Point([2, 0]), 3)
    assert_allclose(beampoly(beam, tri), 4, rtol=1e-6)


# Vertical intersection.
def test_beamcirc_vertical_intersection():
    circ = Circle(Point([0, 0]), 1)
    beam = Probe(Point([-1, -1]), Point([1, 1]), 1)
    assert_allclose(beamcirc(beam, circ), 1.91322295498, rtol=1e-6)


def test_beampoly_vertical_intersection():
    tri = Rectangle(Point([0, 0.5]), side_lengths=[10, 1])
    beam = Probe(Point([0, -1]), Point([0, 1]), 1)
    assert_allclose(beampoly(beam, tri), 1, rtol=1e-6)


def test_sinogram(SIZE=32):
    """Compare sinogram of XDesignDefault with a reference."""
    # Create the phantom
    phan = xd.XDesignDefault()
    # Plot it for debugging purposes
    plt.figure()
    xd.sidebyside(phan, SIZE)
    # Generate the scanning trajectory
    theta, h = xd.raster_scan2D(SIZE, SIZE)
    # Create a probe
    prb = xd.Probe(size=1/SIZE)
    # Meausure the phantom with the probe
    sino = prb.measure(phan, theta, h)
    sino = -np.log(sino)
    # Plot the sinogram for debugging
    plt.figure()
    plt.imshow(sino, origin='lower')
    # Load the reference from file
    ref_file = os.path.join(os.path.dirname(__file__), 'test_sinogram.npy')
    if not os.path.isfile(ref_file):
        ImportError('sinogram reference not found; use test_sinogram.ipynb' +
                    'to generate it')
    sino_reference = np.load(ref_file)
    # assert that they are equal
    np.testing.assert_allclose(sino, sino_reference)


if __name__ == '__main__':
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    test_sinogram()
    plt.show()
