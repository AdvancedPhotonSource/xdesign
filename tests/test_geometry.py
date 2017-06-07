#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2016, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from xdesign.geometry import *
from xdesign.acquisition import beamcirc, beampoly
from xdesign.acquisition import *
from numpy.testing import assert_allclose, assert_raises, assert_equal
import numpy as np


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


# Transformations

def rotate_around_self():
    P0 = Point([0, 0])
    origin = P0
    P1 = rotate(P0, np.pi/2, origin)
    assert_allclose([P1.x, P1.y], [0, 0])


def rotate_around_other():
    P0 = Point([0, 0])
    origin = Point([1, 0])
    P1 = rotate(P0, np.pi/2, origin)
    assert_allclose([P1.x, P1.y], [1, -1])


def test_Point_rotate_around_self():
    P0 = Point([0, 0])
    origin = P0
    P0.rotate(np.pi/2, origin)
    assert_allclose([P0.x, P0.y], [0, 0])


def test_Point_rotate_around_other():
    P0 = Point([0, 0])
    origin = Point([1, 0])
    P0.rotate(np.pi/2, origin)
    assert_allclose([P0.x, P0.y], [1, -1])

    P0 = Point([0, 0])
    origin = Point([0, 1])
    P0.rotate(np.pi/2, origin)
    assert_allclose([P0.x, P0.y], [1, 1])

    P0 = Point([0, 0])
    origin = Point([-1, 0])
    P0.rotate(np.pi/2, origin)
    assert_allclose([P0.x, P0.y], [-1, 1])

    P0 = Point([0, 0])
    origin = Point([-1, -1])
    P0.rotate(np.pi/2, origin)
    assert_allclose([P0.x, P0.y], [-2, 0])


def test_translate():
    P0 = Point([0, 0])
    P0.translate([2.3, 4.5])
    assert_equal([P0.x, P0.y], [2.3, 4.5])


# Nonintersecting beams

def test_beamcirc_nonintersecting_top():
    circ = Circle(Point([0, 3]), 1)
    beam = Beam(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 0., rtol=1e-6)


def test_beamcirc_nonintersecting_bottom():
    circ = Circle(Point([0, -3]), 1)
    beam = Beam(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 0., rtol=1e-6)


def test_beampoly_nonintersecting_top():
    tri = Triangle(Point([0, 1]), Point([1, -1]), Point([-1, -1]))
    beam = Beam(Point([-2, 2]), Point([2, 2]), 2)
    assert_allclose(beampoly(beam, tri), 0., rtol=1e-6)


def test_beampoly_nonintersecting_bottom():
    tri = Triangle(Point([0, 1]), Point([1, -1]), Point([-1, -1]))
    beam = Beam(Point([-2, -2]), Point([2, -2]), 2)
    assert_allclose(beampoly(beam, tri), 0., rtol=1e-6)


# Partial intersections

def test_beamcirc_intersecting_partially_from_top_outside_center():
    circ = Circle(Point([0, 1.5]), 1)
    beam = Beam(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 0.614184849304, rtol=1e-6)


def test_beamcirc_intersecting_partially_from_bottom_outside_center():
    circ = Circle(Point([0, -1.5]), 1)
    beam = Beam(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 0.614184849304, rtol=1e-6)


def test_beamcirc_intersecting_partially_from_top_inside_center():
    circ = Circle(Point([0, 0.5]), 1)
    beam = Beam(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 2.52740780429, rtol=1e-6)


def test_beamcirc_intersecting_partially_from_bottom_inside_center():
    circ = Circle(Point([0, -0.5]), 1)
    beam = Beam(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 2.52740780429, rtol=1e-6)


def test_beampoly_intersecting_partially_from_top():
    tri = Rectangle(Point([0, 0]), Point([1, 0]), Point([1, 1]), Point([0, 1]))
    beam = Beam(Point([-2, 1]), Point([2, 1]), 1)
    assert_allclose(beampoly(beam, tri), 1/2, rtol=1e-6)


def test_beampoly_intersecting_partially_from_bottom():
    tri = Rectangle(Point([0, 0]), Point([1, 0]), Point([1, 1]), Point([0, 1]))
    beam = Beam(Point([-2, 0]), Point([2, 0]), 1)
    assert_allclose(beampoly(beam, tri), 1/2, rtol=1e-6)


# Full intersections

def test_beamcirc_intersecting_fully_from_top_outside_center():
    circ = Circle(Point([0, 1.5]), 3)
    beam = Beam(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 10.0257253792, rtol=1e-6)


def test_beamcirc_intersecting_fully_from_bottom_outside_center():
    circ = Circle(Point([0, -1.5]), 3)
    beam = Beam(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 10.0257253792, rtol=1e-6)


def test_beamcirc_intersecting_fully_from_top_inside_center():
    circ = Circle(Point([0, 0.5]), 3)
    beam = Beam(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 11.5955559562, rtol=1e-6)


def test_beamcirc_intersecting_fully_from_bottom_inside_center():
    circ = Circle(Point([0, -0.5]), 3)
    beam = Beam(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 11.5955559562, rtol=1e-6)


def test_beamcirc_intersecting_fully():
    circ = Circle(Point([0, 0]), 1)
    beam = Beam(Point([-2, 0]), Point([2, 0]), 2)
    assert_allclose(beamcirc(beam, circ), 3.14159265359, rtol=1e-6)


def test_beampoly_intersecting_fully():
    tri = Rectangle(Point([-1, -1]), Point([1, -1]), Point([1, 1]), Point([-1, 1]))
    beam = Beam(Point([-2, 0]), Point([2, 0]), 3)
    assert_allclose(beampoly(beam, tri), 4, rtol=1e-6)


# Vertical intersection.

def test_beamcirc_vertical_intersection():
    circ = Circle(Point([0, 0]), 1)
    beam = Beam(Point([-1, -1]), Point([1, 1]), 1)
    assert_allclose(beamcirc(beam, circ), 1.91322295498, rtol=1e-6)


def test_beampoly_vertical_intersection():
    tri = Rectangle(Point([-5, 0]), Point([5, 0]), Point([5, 1]), Point([-5, 1]))
    beam = Beam(Point([0, -1]), Point([0, 1]), 1)
    assert_allclose(beampoly(beam, tri), 1, rtol=1e-6)


# Line

def test_Line_slope_vertical():
    line = Line(Point([0, -1]), Point([0, 1]))
    assert_allclose(line.slope, np.inf, rtol=1e-6)


def test_Line_yintercept_vertical():
    line = Line(Point([0, -1]), Point([0, 1]))
    assert_allclose(line.yintercept, np.inf, rtol=1e-6)


def test_Line_slope():
    line = Line(Point([-1, 0]), Point([1, 2]))
    assert_allclose(line.slope, 1, rtol=1e-6)


def test_Line_yintercept():
    line = Line(Point([-1, 0]), Point([1, 2]))
    assert_allclose(line.yintercept, 1, rtol=1e-6)


def test_Line_same_points():
    assert_raises(ValueError, Line, Point([1, 2]), Point([1, 2]))


# Circle

def test_Circle_area():
    circle = Circle(Point([0, 0]), 1)
    assert_allclose(circle.area, 3.14159265359, rtol=1e-6)

    negcircle = -circle
    assert_allclose(circle.area, 3.14159265359, rtol=1e-6)
    assert_allclose(negcircle.area, -3.14159265359, rtol=1e-6)


def test_Mesh_area():
    p5 = Point([0, 0])
    p1 = Point([1, 1])
    p4 = Point([1, -1])
    p3 = Point([-1, -1])
    p2 = Point([-1, 1])
    m = Mesh()
    assert_equal(m.area, 0)
    m.append(Triangle(p5, p1, p2))
    m.append(Triangle(p5, p2, p3))
    m.append(Triangle(p5, p3, p4))
    m.append(Triangle(p5, p4, p1))
    assert_equal(m.area, 4)


def test_Mesh_center():
    p5 = Point([0, 0])
    p1 = Point([1, 1])
    p4 = Point([1, -1])
    p3 = Point([-1, -1])
    p2 = Point([-1, 1])
    m = Mesh()
    assert_equal(m.center, Point([0, 0]))

    m.append(Triangle(p5, p1, p2))
    m.append(Triangle(p5, p2, p3))
    m.append(Triangle(p5, p3, p4))
    m.append(Triangle(p5, p4, p1))
    assert_equal(m.center, Point([0, 0]))

    m.pop()
    m.pop()
    m.pop()
    m.pop()
    assert_equal(m.center, Point([0, 0]))


def contains_full_overlap(A, B):
    """Tests the contains function for two entities which are arranged such
    that A is a subset of B and the edges of A and B do not touch."""
    # A = Circle(Point([0, 0]), 0.5)
    # B = Circle(Point([0, 0.1]), 2)

    assert not A.contains(B)
    assert B.contains(A)

    assert not (-A).contains(B)
    assert not B.contains(-A)

    assert not A.contains(-B)
    assert not (-B).contains(A)

    assert (-A).contains(-B)
    assert not (-B).contains(-A)


def contains_partial_overlap(A, B):
    """Tests the contains function for two entities which are arranged such
    that A is a partial subset of B i.e. the edges intersect at least once."""
    # A = Circle(Point([0, 0]), 0.5)
    # B = Circle(Point([0, 1]), 0.5)

    assert not A.contains(B)
    assert not B.contains(A)

    assert not (-A).contains(B)
    assert not B.contains(-A)

    assert not A.contains(-B)
    assert not (-B).contains(A)

    assert not (-A).contains(-B)
    assert not (-B).contains(-A)


def contains_no_overlap(A, B):
    """Tests the contains function for two entities which are arranged such
    that A intersect B is the empty set."""
    # A = Circle(Point([0, 0]), 0.5)
    # B = Circle(Point([0, 3]), 0.5)

    assert not A.contains(B)
    assert not B.contains(A)

    assert (-A).contains(B)
    assert not B.contains(-A)

    assert not A.contains(-B)
    assert (-B).contains(A)

    assert not (-A).contains(-B)
    assert not (-B).contains(-A)


def test_Circle_contains():
    A = Circle(Point([0, 0]), 0.5)
    Bf = Circle(Point([0, 0.1]), 1.5)
    Bp = Circle(Point([0.5, 0.5]), 0.5)
    Bn = Circle(Point([0.5, 3]), 0.5)

    contains_full_overlap(A, Bf)
    contains_partial_overlap(A, Bp)
    contains_no_overlap(A, Bn)

    Bf = Square(Point([0, 0.1]), 3)
    Bp = Square(Point([0.5, 0.5]), 1)
    Bn = Square(Point([0.5, 3]), 1)

    contains_full_overlap(A, Bf)
    contains_partial_overlap(A, Bp)
    contains_no_overlap(A, Bn)


def test_Polygon_contains():
    A = Square(Point([0, 0]), 1)
    Bf = Square(Point([0, 0.1]), 3)
    Bp = Square(Point([0.5, 0.5]), 1)
    Bn = Square(Point([0.5, 3]), 1)

    contains_full_overlap(A, Bf)
    contains_partial_overlap(A, Bp)
    contains_no_overlap(A, Bn)

    Bf = Circle(Point([0, 0.1]), 1.5)
    Bp = Circle(Point([0.5, 0.5]), 0.5)
    Bn = Circle(Point([0.5, 3]), 0.5)

    contains_full_overlap(A, Bf)
    contains_partial_overlap(A, Bp)
    contains_no_overlap(A, Bn)


def test_Mesh_contains():
    p0 = Point([0, 0])
    p1 = Point([0, 1])
    p2 = Point([0, 3])
    circle0 = -Square(Point([0, 0]), 1)
    circle1 = Square(Point([0, 0]), 2)

    assert not circle1.contains(circle0)
    assert (circle0).contains(-circle1)
    assert circle1.contains(p0)

    assert not circle0.contains(circle1)
    assert not circle0.contains(p0)

    mesh0 = Mesh(faces=[circle1, circle0])

    assert not mesh0.contains(p0)
    assert mesh0.contains(p1)
    assert not mesh0.contains(p2)


if __name__ == '__main__':
    import nose
    nose.runmodule(exit=False)
