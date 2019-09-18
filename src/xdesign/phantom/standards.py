"""Defines an object for simulating X-ray phantoms.

.. moduleauthor:: Daniel J Ching <carterbox@users.noreply.github.com>
"""
__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'XDesignDefault',
    'HyperbolicConcentric',
    'DynamicRange',
    'DogaCircles',
    'SlantedSquares',
    'UnitCircle',
    'SiemensStar',
]

from copy import deepcopy
import logging
import pickle
import warnings

import numpy as np

from xdesign.constants import PI
from xdesign.geometry import *
from xdesign.material import *
from xdesign.phantom.phantom import *

logger = logging.getLogger(__name__)


class XDesignDefault(Phantom):
    """Generates a Phantom for internal testing of XDesign.

    The default phantom is: (1) nested, it contains phantoms within phantoms;
    (2) geometrically simple, the sinogram can be verified visually;
    and (3) representative, it contains the three main geometric elements:
    circle, polygon, and mesh.
    """

    def __init__(self):
        super(XDesignDefault, self).__init__(
            geometry=Circle(Point([0.5, 0.5]), radius=0.5),
            material=SimpleMaterial(0.0)
        )

        # define the points of the mesh
        a = Point([0.6, 0.6])
        b = Point([0.6, 0.4])
        c = Point([0.8, 0.4])
        d = (a + c) / 2
        e = (a + b) / 2

        t0 = Triangle(deepcopy(b), deepcopy(c), deepcopy(d))

        # construct and reposition the mesh
        m0 = Mesh()
        m0.append(Triangle(deepcopy(a), deepcopy(e), deepcopy(d)))
        m0.append(Triangle(deepcopy(b), deepcopy(d), deepcopy(e)))

        # define the circles
        m1 = Mesh()
        m1.append(Circle(Point([0.3, 0.5]), radius=0.1))
        m1.append(-Circle(Point([0.3, 0.5]), radius=0.02))

        # construct Phantoms
        self.append(
            Phantom(
                children=[
                    Phantom(geometry=t0, material=SimpleMaterial(0.5)),
                    Phantom(geometry=m0, material=SimpleMaterial(0.5))
                ]
            )
        )
        self.append(Phantom(geometry=m1, material=SimpleMaterial(1.0)))
        self.translate([-0.5, -0.5])


class HyperbolicConcentric(Phantom):
    """Generates a series of cocentric alternating black and white circles whose
    radii are changing at a parabolic rate. These line spacings cover a range
    of scales and can be used to estimate the Modulation Transfer Function. The
    radii change according to this function: r(n) = r0*(n+1)^k.

    Attributes
    ----------
    radii : list
        The list of radii of the circles
    widths : list
        The list of the widths of the bands
    """

    def __init__(self, min_width=0.1, exponent=1 / 2):
        """
        Parameters
        ----------
        min_width : scalar
            The radius of the smallest ring in the series.
        exponent : scalar
            The exponent in the function r(n) = r0*(n+1)^k.
        """

        super(HyperbolicConcentric, self).__init__()
        center = Point([0.0, 0.0])
        Nmax_rings = 512

        radii = [0]
        widths = [min_width]
        for ring in range(0, Nmax_rings):
            radius = min_width * np.power(ring + 1, exponent)
            if radius > 0.5 and ring % 2:
                break

            self.append(
                Phantom(
                    geometry=Circle(center, radius),
                    material=SimpleMaterial((-1.)**(ring % 2))
                )
            )
            # record information about the rings
            widths.append(radius - radii[-1])
            radii.append(radius)

        self.children.reverse()  # smaller circles on top
        self.radii = radii
        self.widths = widths


class DynamicRange(Phantom):
    """Generates a phantom of randomly placed circles for determining dynamic
    range.

    Parameters
    -------------
    steps : scalar, optional
        The orders of magnitude (base 2) that the colors of the circles cover.
    jitter : bool, optional
        True : circles are placed in a jittered grid
        False : circles are randomly placed
    shape : string, optional
    """

    def __init__(
        self,
        steps=10,
        jitter=True,
        geometry=Square(center=Point([0.5, 0.5]), side_length=1)
    ):
        super(DynamicRange, self).__init__(geometry=geometry)

        # determine the size and and spacing of the circles around the box.
        spacing = 1.0 / np.ceil(np.sqrt(steps))
        radius = spacing / 4

        colors = [2.0**j for j in range(0, steps)]
        np.random.shuffle(colors)

        if jitter:
            # generate grid
            _x = np.arange(0, 1, spacing) + spacing / 2
            px, py, = np.meshgrid(_x, _x)
            px = np.ravel(px)
            py = np.ravel(py)

            # calculate jitters
            jitters = 2 * radius * (np.random.rand(2, steps) - 0.5)

            # place the circles
            for i in range(0, steps):
                center = Point([px[i] + jitters[0, i], py[i] + jitters[1, i]])
                self.append(
                    Phantom(
                        geometry=Circle(center, radius),
                        material=SimpleMaterial(colors[i])
                    )
                )
        else:
            # completely random
            for i in range(0, steps):
                if 1 > self.sprinkle(
                    1,
                    radius,
                    gap=radius * 0.9,
                    material=SimpleMaterial(colors[i])
                ):
                    None
                    # TODO: ensure that all circles are placed
        self.translate([-0.5, -0.5])


class DogaCircles(Phantom):
    """Rows of increasingly smaller circles. Initally arranged in an ordered
    Latin square, the inital arrangement can be randomly shuffled.

    Attributes
    ----------
    radii : ndarray
        radii of circles
    x : ndarray
        x position of circles
    y : ndarray
        y position of circles
    """

    # IDEA: Use method in this reference to calculate uniformly distributed
    # latin squares.
    # DOI: 10.1002/(SICI)1520-6610(1996)4:6<405::AID-JCD3>3.0.CO;2-J
    def __init__(self, n_sizes=5, size_ratio=0.5, n_shuffles=5):
        """
        Parameters
        ----------
        n_sizes : int
            number of different sized circles
        size_ratio : scalar
            the nth size / the n-1th size
        n_shuffles : int
            The number of times to shuffles the latin square
        """
        super(DogaCircles, self).__init__(
            geometry=Circle(center=Point([0.5, 0.5]), radius=0.5)
        )

        n_sizes = int(n_sizes)
        if n_sizes <= 0:
            raise ValueError('There must be at least one size.')
        if size_ratio > 1 or size_ratio <= 0:
            raise ValueError('size_ratio should be <= 1 and > 0.')
        n_shuffles = int(n_shuffles)
        if n_shuffles < 0:
            raise ValueError('Cant shuffle a negative number of times')

        # Seed a latin square, use integers to prevent rounding errors
        top_row = np.array(range(0, n_sizes), dtype=int)
        rowsum = np.sum(top_row)
        lsquare = np.empty([n_sizes, n_sizes], dtype=int)
        for i in range(0, n_sizes):
            lsquare[:, i] = np.roll(top_row, i)

        # Choose a row or column shuffle sequence
        sequence = np.random.randint(0, 2, n_shuffles)

        # Shuffle the square
        for dim in sequence:
            lsquare = np.rollaxis(lsquare, dim, 0)
            np.random.shuffle(lsquare)

        # Assert that it is still a latin square.
        for i in range(0, n_sizes):
            assert np.sum(lsquare[:, i]) == rowsum, \
                "Column {0} is {1} and should be {2}".format(i, np.sum(
                                                        lsquare[:, i]), rowsum)
            assert np.sum(lsquare[i, :]) == rowsum, \
                "Column {0} is {1} and should be {2}".format(i, np.sum(
                                                        lsquare[i, :]), rowsum)

        # Draw it
        period = (np.arange(0, n_sizes) / n_sizes + 1 / (2 * n_sizes)) * 0.7
        _x, _y = np.meshgrid(period, period)
        radii = (1 - 1e-10) / (2 * n_sizes) * size_ratio**lsquare * 0.7
        _x += (1 - 0.7) / 2
        _y += (1 - 0.7) / 2

        for (k, x, y) in zip(radii.flatten(), _x.flatten(), _y.flatten()):
            self.append(
                Phantom(
                    geometry=Circle(Point([x, y]), radius=k),
                    material=SimpleMaterial(1.0)
                )
            )

        self.radii = radii
        self.x = _x
        self.y = _y
        self.translate([-0.5, -0.5])


class SlantedSquares(Phantom):
    """Generates a collection of slanted squares. Squares are arranged in
    concentric circles such that the space between squares is at least gap. The
    size of the squares is adaptive such that they all remain within the unit
    circle.

    Attributes
    ----------
    angle : scalar
        the angle of slant in radians
    count : scalar
        the total number of squares
    gap : scalar
        the minimum space between squares
    side_length : scalar
        the size of the squares
    squares_per_level : list
        the number of squares at each level
    radius_per_level : list
        the radius at each level
    n_levels : scalar
        the number of levels
    """

    def __init__(self, count=10, angle=5 / 360 * 2 * PI, gap=0):
        super(SlantedSquares, self).__init__()
        if count < 1:
            raise ValueError("There must be at least one square.")

        # approximate the max diameter from total area available
        d_max = np.sqrt(PI / 4 / (2 * count))

        if 1 < count and count < 5:
            # bump all the squares to the 1st ring and calculate sizes
            # as if there were 5 total squares
            pass

        while True:
            squares_per_level = [1]
            radius_per_level = [0]
            remaining = count - 1
            n_levels = 1
            while remaining > 0:
                # calculate next level capacity
                radius_per_level.append(
                    radius_per_level[n_levels - 1] + d_max + gap
                )
                this_circumference = PI * 2 * radius_per_level[n_levels]
                this_capacity = this_circumference // (d_max + gap)

                # assign squares to levels
                if remaining - this_capacity >= 0:
                    squares_per_level.append(this_capacity)
                    remaining -= this_capacity
                else:
                    squares_per_level.append(remaining)
                    remaining = 0
                n_levels += 1
                assert (remaining >= 0)

            # Make sure squares will not be outside the phantom, else
            # decrease diameter by 5%
            if radius_per_level[-1] < (0.5 - d_max / 2 - gap):
                break
            d_max *= 0.95

        assert (len(squares_per_level) == len(radius_per_level))

        # determine center positions of squares
        x, y = np.array([]), np.array([])
        for level in range(0, n_levels):
            radius = radius_per_level[level]
            thetas = (((
                np.arange(0, squares_per_level[level]) /
                squares_per_level[level]
            ) + 1 / (squares_per_level[level] * 2)) * 2 * PI)
            x = np.concatenate((x, radius * np.cos(thetas)))
            y = np.concatenate((y, radius * np.sin(thetas)))

        # move to center of phantom.
        x += 0.5
        y += 0.5

        # add the squares to the phantom
        side_length = d_max / np.sqrt(2)
        for i in range(0, x.size):
            center = Point([x[i], y[i]])
            s = Square(center=center, side_length=side_length)
            s.rotate(angle, center)
            self.append(Phantom(geometry=s, material=SimpleMaterial(1)))

        self.angle = angle
        self.count = count
        self.gap = gap
        self.side_length = side_length
        self.squares_per_level = squares_per_level
        self.radius_per_level = radius_per_level
        self.n_levels = n_levels
        self.translate([-0.5, -0.5])


class UnitCircle(Phantom):
    """Generates a phantom with a single circle in its center."""

    def __init__(self, radius=0.5, material=SimpleMaterial(1.0)):
        super(UnitCircle, self).__init__(
            geometry=Circle(Point([0.0, 0.0]), radius), material=material
        )


class SiemensStar(Phantom):
    """Generates a Siemens star.

    Attributes
    ----------
    center: Point
        The center of the Siemens Star.
    n_sectors: int >= 2
        The number of spokes/blades on the star.
    radius: scalar > 0
        The radius of the circle inscribing the Siemens Star.
    ratio : scalar
        The spatial frequency times the proportional radius. e.g to get the
        frequency, f, divide this ratio by some fraction of the maximum radius:
        f = ratio/radius_fraction.
        .. deprecated:: 0.5
            Use :func:`SiemensStar.get_frequency` or
            :func:`SiemensStar.get_radius` instead.

    .. versionchanged 0.5
        The `n_sectors` parameter was changed to count only the material as
        spokes instead of both the material and the space between. This allows
        evenly spaced odd numbers of spokes.

    """

    def __init__(self, n_sectors=2, center=Point([0.0, 0.0]), radius=0.5):
        """See help(SiemensStar) for more info."""
        super(SiemensStar, self).__init__()
        if n_sectors < 2:
            raise ValueError("A Siemens star must have > 1 sector.")
        if radius <= 0:
            raise ValueError("radius must be greater than zero.")
        if not isinstance(center, Point):
            raise TypeError("center must be of type Point!")
        n_points = 2 * n_sectors
        self.ratio = n_sectors / (2 * np.pi * radius)
        self.n_sectors = n_sectors
        # generate an even number of points around the unit circle
        points = []
        for t in np.linspace(0, 2 * np.pi, n_points, endpoint=False):
            x = radius * np.cos(t) + center.x
            y = radius * np.sin(t) + center.y
            points.append(Point([x, y]))
        # connect pairs of points to the center to make triangles
        for i in range(0, n_sectors):
            f = Phantom(
                geometry=Triangle(points[2 * i], points[2 * i + 1], center),
                material=SimpleMaterial(1)
            )
            self.append(f)

    def get_frequency(radius):
        """Return the spatial frequency at the given radius.

        .. versionadded:: 0.5
        """
        return self.n_sectors / (2 * np.pi * radius)

    def get_radius(frequency):
        """Return the radius which provides the given frequency.

        .. versionadded:: 0.5
        """
        return self.n_sectors / (2 * np.pi * frequency)
