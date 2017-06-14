from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt

from numpy.testing import assert_allclose

from xdesign import *
from xdesign.phantom import XDesignDefault
from xdesign.geometry import NCube, NOrthotope, Square
from xdesign.acquisition import beamtope
from xdesign.plot import plot_geometry
from polytope.polytope import _rotate
import xdesign


def test_beampoly_intersecting_partially_from_top():
    tri = NCube(Point([0.5, 0.5, 0]), 1.0)
    beam = Probe(Point([-2, 1, 0]), Point([2, 1, 0]), 1)
    assert_allclose(beamtope(beam, tri), 1/2, rtol=1e-6)


def test_beampoly_intersecting_partially_from_bottom():
    tri = NCube(Point([0.5, 0.5, 0]), 1.0)
    beam = Probe(Point([-2, 0, 0]), Point([2, 0, 0]), 1)
    assert_allclose(beamtope(beam, tri), 1/2, rtol=1e-6)


def test_beampoly_intersecting_fully():
    tri = NCube(Point([0, 0, 0]), 2.0)
    beam = Probe(Point([-2, 0, 0]), Point([2, 0, 0]), 3)
    assert_allclose(beamtope(beam, tri), 8, rtol=1e-6)


def test_beampoly_vertical_intersection():
    tri = NOrthotope(Point([0, 0.5, 0]), [10, 1, 1])
    beam = Probe(Point([0, -1, 0]), Point([0, 1, 0]), 1)
    assert_allclose(beamtope(beam, tri), 1, rtol=1e-6)


def test_probe_circular(N=4):
    beam = Probe(Point([0, 0, -1]), Point([0, 0, 1]), 1, circleapprox=N)
    tri = NCube(Point([0, 0, 0]), 1.0)
    assert_allclose(beamtope(beam, tri), np.pi/4, rtol=1e-6)


def test_plot_polytope():
    tri = NCube(Point([0.5, 0.5, 0]), 0.1)

    tri.rotate(np.pi/4, Point([0.5, 0.5, 0]))

    _, axis = xdesign.plot._make_axis()
    plot_geometry(tri, axis)
    plot_geometry(Square(Point([0.3, 0.3]), 0.1), axis)
    plt.show(block=True)


if __name__ == '__main__':
    test_probe_circular(N=32)
