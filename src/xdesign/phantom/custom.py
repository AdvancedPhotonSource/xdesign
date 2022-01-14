"""Defines an object for simulating X-ray phantoms.

.. moduleauthor:: Daniel J Ching <carterbox@users.noreply.github.com>
"""

__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'Soil',
    'WetCircles',
    'Foam',
    'Softwood',
]

from copy import deepcopy
import itertools
import logging
import pickle
import warnings

import numpy as np
from scipy.spatial import Delaunay

from xdesign.constants import PI
from xdesign.geometry import *
from xdesign.material import *
from xdesign.phantom.phantom import *
from xdesign.phantom.standards import *

logger = logging.getLogger(__name__)


class Foam(UnitCircle):
    """Generates a phantom with structure similar to foam."""

    def __init__(self, size_range=[0.05, 0.01], gap=0, porosity=1):
        super(Foam, self).__init__(radius=0.5, material=SimpleMaterial(1.0))
        if porosity < 0 or porosity > 1:
            raise ValueError('Porosity must be in the range [0,1).')
        self.sprinkle(
            300,
            size_range,
            gap,
            material=SimpleMaterial(-1.0),
            max_density=porosity
        )


class Softwood(Phantom):
    """Generate a Phantom with structure similar to wood.

    Parameters
    ----------
    ringsize : float [cm]
        The thickness of the annual rings in cm.
    latewood_fraction : float
        The volume ratio of latewood cells to earlywood cells
    ray_fraction : float
        The ratio of rows of ray cells to rows of tracheids
    ray_height : float [cm]
        The height of the ray cells
    cell_width, cell_height : float [cm]
        The shape of the earlywood cells
    cell_thickness : float [cm]
        The thickness of the earlywood cell walls
    frame : arraylike [cm]
        A bounding box for the cells
    """

    def __init__(self):
        super(Softwood, self).__init__()

        ring_size = 0.5
        ring_offset = np.random.rand()
        latewood_fraction = 0.35

        ray_fraction = 1 / 8
        ray_height = 0.01
        ray_width = 0.09
        ray_thickness = 0.002

        cell_width, cell_height = 0.03, 0.03
        cell_thickness = 0.004

        frame = np.array([[-0.2, -0.2], [0.2, 0.2]])

        # -------------------
        def five_p():
            return 1 + np.random.normal(scale=0.05)

        atol = 1e-16  # for rounding errors
        cellulose = SimpleMaterial(1)

        x0, y0 = frame[0, 0], frame[0, 1]
        x1, y1 = frame[1, 0], frame[1, 1]

        # Place the cells one by one at (x, y)
        y = y0
        for r in itertools.count():
            # Check that the row is in the frame
            if y + cell_height > y1 and abs(y + cell_height - y1) > atol:
                # Stop if cell reaches out of frame
                break

            # Add random jitter to each row
            x = x0 + cell_width * np.random.normal(scale=0.1)
            if r % 2 == 1:
                # Offset odd number rows by 1/2 cell width
                x += cell_width / 2

            # Decide whether to make a ray cell
            if np.random.rand() < ray_fraction:
                is_ray = True
            else:
                is_ray = False

            for c in itertools.count():
                ring_progress = ((x + ring_offset) / ring_size) % 1

                if x < x0 and abs(x - x0) > atol:
                    # skip first cell if jittered outside the frame
                    x += cell_width

                if is_ray:
                    cell = WoodCell(
                        corner=Point([x, y]),
                        material=cellulose,
                        width=ray_width * five_p(),
                        height=ray_height,
                        wall_thickness=ray_thickness * five_p()
                    )

                else:  # not ray cells
                    if ring_progress < 1 - latewood_fraction:
                        # earlywood
                        dw, dt = 1, 1
                    else:
                        # transition to latewood
                        dw = 0.6
                        dt = 1.5

                    cell = WoodCell(
                        corner=Point([x, y]),
                        material=cellulose,
                        width=cell_width * dw * five_p(),
                        height=cell_height,
                        wall_thickness=cell_thickness * dt * five_p()
                    )
                self.append(cell)

                x += cell.width

                if x + cell.width > x1 and abs(x + cell.width - x1) > atol:
                    # Stop if cell reaches out of frame
                    break

            y += cell.height


class WoodCell(Phantom):
    """Generate a Phantom with structure similar to a single wood cell.

    A wood cell has two parts: the lumen which is the empty center area of the
    cell and the cell wall substance which is generally hexagonal.
    """

    def __init__(
        self,
        corner=Point([0.5, 0.5]),
        width=0.003,
        height=0.003,
        wall_thickness=0.0008,
        material=None
    ):
        super(WoodCell, self).__init__()

        p1 = deepcopy(corner) + Point([width / 2, height / 2])
        cell_wall = Rectangle(p1, [width, height])

        wt = wall_thickness
        p1 = deepcopy(p1)
        lumen = -Rectangle(p1, [width - 2 * wt, height - 2 * wt])

        self._geometry = Mesh(faces=[cell_wall, lumen])
        self.material = material
        self.height = height
        self.width = width
        # self.append(center)


class Soil(UnitCircle):
    """Generates a phantom with structure similar to soil.

    References
    -----------
    Schlüter, S., Sheppard, A., Brown, K., & Wildenschild, D. (2014). Image
    processing of multiphase images obtained via X‐ray microtomography: a
    review. Water Resources Research, 50(4), 3615-3639.
    """

    def __init__(self, porosity=0.412):
        super(Soil, self).__init__(radius=0.5, material=SimpleMaterial(0.5))
        self.sprinkle(
            30, [0.1, 0.03],
            0,
            material=SimpleMaterial(0.5),
            max_density=1 - porosity
        )
        # use overlap to approximate area opening transform because opening is
        # not discrete
        self.sprinkle(100, 0.02, 0.01, material=SimpleMaterial(-.25))


class WetCircles(UnitCircle):
    def __init__(self):
        super(WetCircles, self).__init__(
            radius=0.5, material=SimpleMaterial(0.5)
        )
        porosity = 0.412
        np.random.seed(0)

        self.sprinkle(
            30, [0.1, 0.03],
            0.005,
            material=SimpleMaterial(0.5),
            max_density=1 - porosity
        )

        pairs = [(23, 12), (12, 19), (29, 11), (22, 5), (1, 3), (21, 9), (8, 2),
                 (2, 27)]
        for p in pairs:
            A = self.children[p[0] - 1].geometry
            B = self.children[p[1] - 1].geometry

            thetaA = [PI / 2, 10]
            thetaB = [PI / 2, 10]

            mesh = wet_circles(A, B, thetaA, thetaB)

            self.append(Phantom(geometry=mesh, material=SimpleMaterial(-.25)))


def wet_circles(A, B, thetaA, thetaB):
    """Generates a mesh that wets the surface of circles A and B.

    Parameters
    -------------
    A,B : Circle
    theta : list
        the number of radians that the wet covers and number of the points on
        the surface range
    """

    vector = B.center - A.center
    if vector.x > 0:
        angleA = np.arctan(vector.y / vector.x)
        angleB = PI + angleA
    else:
        angleB = np.arctan(vector.y / vector.x)
        angleA = PI + angleB
    # print(vector)
    rA = A.radius
    rB = B.radius

    points = []
    for t in ((np.arange(0, thetaA[1]) / (thetaA[1] - 1) - 0.5) * thetaA[0] +
              angleA):

        x = rA * np.cos(t) + A.center.x
        y = rA * np.sin(t) + A.center.y
        points.append([x, y])

    mid = len(points)
    for t in ((np.arange(0, thetaB[1]) / (thetaB[1] - 1) - 0.5) * thetaB[0] +
              angleB):

        x = rB * np.cos(t) + B.center.x
        y = rB * np.sin(t) + B.center.y
        points.append([x, y])

    points = np.array(points)

    # Triangulate the polygon
    tri = Delaunay(points)

    # Remove extra triangles
    # print(tri.simplices)
    mask = np.sum(tri.simplices < mid, 1)
    mask = np.logical_and(mask < 3, mask > 0)
    tri.simplices = tri.simplices[mask, :]
    # print(tri.simplices)

    m = Mesh()
    for t in tri.simplices:
        m.append(
            Triangle(
                Point([points[t[0], 0], points[t[0], 1]]),
                Point([points[t[1], 0], points[t[1], 1]]),
                Point([points[t[2], 0], points[t[2], 1]])
            )
        )

    return m
