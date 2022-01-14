"""Define two dimensional geometric entities."""

__author__ = "Daniel Ching, Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'Curve',
    'Circle',
    'Polygon',
    'RegularPolygon',
    'Triangle',
    'Rectangle',
    'Square',
    'Mesh',
]

from copy import deepcopy
import logging
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from cached_property import cached_property

from xdesign.geometry.entity import *
from xdesign.geometry.line import *
from xdesign.geometry.point import *

logger = logging.getLogger(__name__)


class Curve(Entity):
    """The base class for closed manifolds defined by a single equation. e.g.
    :class:`.Circle`, :class:`.Sphere`, or :class:`.Torus`.

    Attributes
    ----------
    center : Point
    """

    def __init__(self, center):
        if not isinstance(center, Point):
            raise TypeError("center must be a Point.")
        super(Curve, self).__init__()
        self.center = center

    def __repr__(self):
        return "{}(center={})".format(type(self).__name__, repr(self.center))

    def translate(self, vector):
        """Translates the Curve along a vector."""
        if not isinstance(vector, (Point, list, np.array)):
            raise TypeError("vector must be point, list, or array.")
        self.center.translate(vector)

    def rotate(self, theta, point=None, axis=None):
        """Rotates the Curve by theta radians around an axis which passes
        through a point radians."""
        self.center.rotate(theta, point, axis)


class Superellipse(Curve):
    """A Superellipse in 2D cartesian space.

    Attributes
    ----------
    center : Point
    a : scalar
    b : scalar
    n : scalar
    """

    def __init__(self, center, a, b, n):
        super(Superellipse, self).__init__(center)
        self.a = float(a)
        self.b = float(b)
        self.n = float(n)

    def __repr__(self):
        return "Superellipse(center={}, a={}, b={}, n={})".format(
            repr(self.center), repr(self.a), repr(self.b), repr(self.n)
        )

    @property
    def list(self):
        """Return list representation."""
        return [self.center.x, self.center.y, self.a, self.b, self.n]

    def scale(self, val):
        """Scale."""
        self.a *= val
        self.b *= val


class Ellipse(Superellipse):
    """Ellipse in 2-D cartesian space.

    Attributes
    ----------
    center : Point
    a : scalar
    b : scalar
    """

    def __init__(self, center, a, b):
        super(Ellipse, self).__init__(center, a, b, 2)

    def __repr__(self):
        return "Ellipse(center={}, a={}, b={})".format(
            repr(self.center), repr(self.a), repr(self.b)
        )

    @property
    def list(self):
        """Return list representation."""
        return [self.center.x, self.center.y, self.a, self.b]

    @property
    def area(self):
        """Return area."""
        return np.pi * self.a * self.b

    def scale(self, val):
        """Scale."""
        self.a *= val
        self.b *= val


class Circle(Curve):
    """Circle in 2D cartesian space.

    Attributes
    ----------
    center : Point
        The center point of the circle.
    radius : scalar
        The radius of the circle.
    sign : int (-1 or 1)
        The sign of the area
    """

    def __init__(self, center, radius, sign=1):
        super(Circle, self).__init__(center)
        self.radius = float(radius)
        self.sign = sign
        self._dim = 2

    def __repr__(self):
        return "Circle(center={}, radius={}, sign={})".format(
            repr(self.center), repr(self.radius), repr(self.sign)
        )

    def __str__(self):
        """Return the analytical equation."""
        return "(x-%s)^2 + (y-%s)^2 = %s^2" % (
            self.center.x, self.center.y, self.radius
        )

    def __eq__(self, circle):
        return ((self.x, self.y,
                 self.radius) == (circle.x, circle.y, circle.radius))

    def __neg__(self):
        copE = deepcopy(self)
        copE.sign = -copE.sign
        return copE

    @property
    def list(self):
        """Return list representation for saving to files."""
        return [self.center.x, self.center.y, self.radius]

    @property
    def circumference(self):
        """Returns the circumference."""
        return 2 * np.pi * self.radius

    @property
    def diameter(self):
        """Returns the diameter."""
        return 2 * self.radius

    @property
    def area(self):
        """Return the area."""
        return self.sign * np.pi * self.radius**2

    @property
    def patch(self):
        """Returns a matplotlib patch."""
        return plt.Circle((self.center.y, self.center.x), self.radius)

    @property
    def bounding_box(self):
        """Return the axis-aligned bounding box as two numpy vectors."""
        xmin = np.array(self.center._x - self.radius)
        xmax = np.array(self.center._x + self.radius)

        return xmin, xmax

    # def scale(self, val):
    #     """Scale."""
    #     raise NotImplementedError
    #     self.center.scale(val)
    #     self.rad *= val

    def contains(self, other):
        """Return whether `other` is a proper subset.

        Return one boolean for all geometric entities. Return an array of
        boolean for array input.
        """
        if isinstance(other, Point):
            x = other._x
        elif isinstance(other, np.ndarray):
            x = other
        elif isinstance(other, Mesh):
            for face in other.faces:
                if not self.contains(face) and face.sign == 1:
                    return False
            return True
        else:
            if self.sign == 1:
                if other.sign == -1:
                    # Closed shape cannot contain infinite one
                    return False
                else:
                    assert other.sign == 1
                    # other is within A
                    if isinstance(other, Circle):
                        return (
                            other.center.distance(self.center) + other.radius <
                            self.radius
                        )
                    elif isinstance(other, Polygon):
                        x = _points_to_array(other.vertices)
                        return np.all(self.contains(x))

            elif self.sign == -1:
                if other.sign == 1:
                    # other is outside A and not around
                    if isinstance(other, Circle):
                        return (
                            other.center.distance(self.center) - other.radius >
                            self.radius
                        )
                    elif isinstance(other, Polygon):
                        x = _points_to_array(other.vertices)
                        return (
                            np.all(self.contains(x))
                            and not other.contains(-self)
                        )

                else:
                    assert other.sign is -1
                    # other is around A
                    if isinstance(other, Circle):
                        return (
                            other.center.distance(self.center) + self.radius <
                            other.radius
                        )
                    elif isinstance(other, Polygon):
                        return (-other).contains(-self)

        x = np.atleast_2d(x)

        if self.sign == 1:
            return np.sum((x - self.center._x)**2, axis=1) < self.radius**2
        else:
            return np.sum((x - self.center._x)**2, axis=1) > self.radius**2


def _points_to_array(points):
    a = np.zeros((len(points), points[0].dim))

    for i in range(len(points)):
        a[i] = points[i]._x

    return np.atleast_2d(a)


class Polygon(Entity):
    """A convex polygon in 2D cartesian space.

    It is defined by a number of distinct vertices of class :class:`.Point`.
    Superclasses include :class:`.Square`, :class:`.Triangle`, etc.

    Attributes
    ----------
    vertices : List of Points
    sign : int (-1 or 1)
        The sign of the area

    Raises
    ------
    ValueError : If the number of vertices is less than three.
    """

    def __init__(self, vertices, sign=1):
        for v in vertices:
            if not isinstance(v, Point):
                raise TypeError("vertices must be of type Point.")
        if len(vertices) < 3:
            raise ValueError("A Polygon has at least three vertices.")
        super(Polygon, self).__init__()
        self.vertices = vertices
        self._dim = vertices[0].dim
        self.sign = sign

    def __repr__(self):
        return "Polygon(vertices={}, sign={})".format(
            repr(self.vertices), repr(self.sign)
        )

    def __str__(self):
        return "{}({})".format(type(self).__name__, str(self.numpy))

    def __neg__(self):
        copE = deepcopy(self)
        copE.sign = -copE.sign
        return copE

    @property
    def numverts(self):
        return len(self.vertices)

    @property
    def list(self):
        """Return list representation."""
        lst = []
        for m in range(self.numverts):
            lst.append(self.vertices[m].list)
        return lst

    @property
    def numpy(self):
        """Return Numpy representation."""
        return _points_to_array(self.vertices)

    @property
    def patch(self):
        """Returns a matplotlib patch."""
        points = self.vertices
        a = np.zeros((len(points), points[0].dim))
        for i in range(len(points)):
            a[i] = np.flip(points[i]._x, 0)
        return plt.Polygon(a)

    # Cached Properties
    @property
    def bounds(self):
        """Returns a 4-tuple (xmin, ymin, xmax, ymax) representing the
        bounding rectangle for the Polygon.
        """
        warnings.warn(
            "Polygon.bounds is deprecated; use Polygon.bounding_box instead.",
            DeprecationWarning)
        xs = [p.x for p in self.vertices]
        ys = [p.y for p in self.vertices]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def bounding_box(self):
        """Return the axis-aligned bounding box as two numpy vectors."""
        xs = [p.x for p in self.vertices]
        ys = [p.y for p in self.vertices]
        return np.array([min(xs), min(ys)]), np.array([max(xs), max(ys)])

    @property
    def edges(self):
        """Return a list of lines connecting the points of the Polygon."""
        edges = []

        for i in range(self.numverts):
            edges.append(
                Segment(
                    self.vertices[i], self.vertices[(i + 1) % self.numverts]
                )
            )

        return edges

    @cached_property
    def area(self):
        """Return the area of the Polygon.

        References
        ----------
        https://en.wikipedia.org/wiki/Shoelace_formula
        https://stackoverflow.com/a/30408825
        """
        a = _points_to_array(self.vertices)
        x = a[:, 0]
        y = a[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @cached_property
    def perimeter(self):
        """Return the perimeter of the Polygon."""
        perimeter = 0
        verts = self.vertices
        points = verts + [verts[0]]
        for m in range(self.numverts):
            perimeter += points[m].distance(points[m + 1])
        return perimeter

    @cached_property
    def center(self):
        """The center of the bounding circle."""
        center = Point(np.zeros(self._dim))
        for v in self.vertices:
            center += v
        return center / self.numverts

    @cached_property
    def radius(self):
        """The radius of the bounding circle."""
        r = 0
        c = self.center
        for m in range(self.numverts):
            r = max(r, self.vertices[m].distance(c))
        return r

    @cached_property
    def half_space(self):
        """Returns the half space polytope respresentation of the polygon."""
        assert (self.dim > 0), self.dim
        A = np.ndarray((self.numverts, self.dim))
        B = np.ndarray(self.numverts)

        for i in range(0, self.numverts):
            edge = Line(
                self.vertices[i], self.vertices[(i + 1) % self.numverts]
            )
            A[i, :], B[i] = edge.standard

            # test for positive or negative side of line
            if self.center._x.dot(A[i, :]) > B[i]:
                A[i, :] = -A[i, :]
                B[i] = -B[i]

        return A, B

    # Methods
    def translate(self, vector):
        """Translates the polygon by a vector."""
        for v in self.vertices:
            v.translate(vector)

        if 'center' in self.__dict__:
            self.center.translate(vector)

        # if 'bounds' in self.__dict__:
        #     self.bounds.translate(vector)

        if 'half_space' in self.__dict__:
            self.half_space = self.half_space.translation(vector)

    def rotate(self, theta, point=None, axis=None):
        """Rotates the Polygon around an axis which passes through a point by
        theta radians."""
        for v in self.vertices:
            v.rotate(theta, point, axis)

        if 'center' in self.__dict__:
            self.center.rotate(theta, point, axis)

        # if 'bounds' in self.__dict__:
        #     self.bounds.rotate(theta, point, axis)

        if 'half_space' in self.__dict__:
            if point is None:
                d = 0
            else:
                d = point._x
            self.half_space = self.half_space.translation(-d)
            self.half_space = self.half_space.rotation(0, 1, theta)
            self.half_space = self.half_space.translation(d)

    def contains(self, other):
        """Return whether this Polygon contains the other."""

        if isinstance(other, Point):
            x = other._x
        elif isinstance(other, np.ndarray):
            x = other
        elif isinstance(other, Mesh):
            for face in other.faces:
                if not self.contains(face) and face.sign == 1:
                    return False
            return True
        else:
            if self.sign == 1:
                if other.sign == -1:
                    # Closed shape cannot contain infinite one
                    return False
                else:
                    assert other.sign == 1
                    # other is within A
                    if isinstance(other, Circle):
                        if self.contains(other.center):
                            for edge in self.edges:
                                if other.center.distance(edge) < other.radius:
                                    return False
                            return True
                        return False
                    elif isinstance(other, Polygon):
                        x = _points_to_array(other.vertices)
                        return np.all(self.contains(x))

            elif self.sign == -1:
                if other.sign == 1:
                    # other is outside A and not around
                    if isinstance(other, Circle):
                        if self.contains(other.center):
                            for edge in self.edges:
                                if other.center.distance(edge) < other.radius:
                                    return False
                            return True and not other.contains(-self)
                        return False
                    elif isinstance(other, Polygon):
                        x = _points_to_array(other.vertices)
                        return (
                            np.all(self.contains(x))
                            and not other.contains(-self)
                        )

                else:
                    assert other.sign is -1
                    # other is around A
                    if isinstance(other, Circle) or isinstance(other, Polygon):
                        return (-other).contains(-self)

        border = Path(self.numpy)

        if self.sign == 1:
            return border.contains_points(np.atleast_2d(x))
        else:
            return np.logical_not(border.contains_points(np.atleast_2d(x)))


class RegularPolygon(Polygon):
    """A regular polygon in 2D cartesian space.

    It is defined by the polynomial center, order, and radius.

    By default (i.e. when the ``angle`` parameter is zero), the regular
    polygon is oriented so that one of the vertices is at coordinates
    :math:`(x + r, x)` where :math:`x` is the x-coordinate of
    ``center`` and :math:`r` = ``radius``. The ``angle`` parameter is
    only meaningful modulo :math:`2\pi /` ``order`` since rotation by
    :math:`2\pi /` ``order`` gives a result equivalent to no rotation.

    Parameters
    ----------
    center : :class:`Point`
        The center of the polygon
    radius : float
        Distance from polygon center to vertices
    order : int
        Order of the polygon (e.g. order 6 is a hexagon).
    angle : float
        Optional rotation angle in radians.
    sign : int (-1 or 1)
        Optional sign of the area (see :class:`Polygon`)
    """

    def __init__(self, center, radius, order, angle=0, sign=1):
        vertex_angles = (np.linspace(0, 2 * np.pi, order, endpoint=False) +
                         angle)
        vertices = [
            Point([radius * np.cos(theta), radius * np.sin(theta)]) + center
            for theta in vertex_angles
        ]
        super(RegularPolygon, self).__init__(vertices, sign=sign)


class Triangle(Polygon):
    """Triangle in 2D cartesian space.

    It is defined by three distinct points.
    """

    def __init__(self, p1, p2, p3):
        super(Triangle, self).__init__([p1, p2, p3])

    def __repr__(self):
        return "Triangle({}, {}, {})".format(
            self.vertices[0], self.vertices[1], self.vertices[2]
        )

    @cached_property
    def center(self):
        center = Point([0, 0])
        for v in self.vertices:
            center += v
        return center / 3

    @cached_property
    def area(self):
        A = self.vertices[0] - self.vertices[1]
        B = self.vertices[0] - self.vertices[2]
        return self.sign * 1 / 2 * np.abs(np.cross([A.x, A.y], [B.x, B.y]))


class Rectangle(Polygon):
    """Rectangle in 2D cartesian space.

    Defined by a point and a vector to enforce perpendicular sides.

    Parameters
    ----------
    side_lengths : array
        The lengths of the sides
    """

    def __init__(self, center, side_lengths):

        s = np.array(side_lengths) / 2
        self.side_lengths = np.array(side_lengths)

        p1 = Point([center.x + s[0], center.y + s[1]])
        p2 = Point([center.x - s[0], center.y + s[1]])
        p3 = Point([center.x - s[0], center.y - s[1]])
        p4 = Point([center.x + s[0], center.y - s[1]])

        super(Rectangle, self).__init__([p1, p2, p3, p4])

    def __repr__(self):
        return "Rectangle({}, {})".format(
            repr(self.center), repr(self.side_lengths.tolist())
        )

    @cached_property
    def area(self):
        return self.sign * (
            self.vertices[0].distance(self.vertices[1]) *
            self.vertices[1].distance(self.vertices[2])
        )


class Square(Rectangle):
    """Square in 2D cartesian space.

    Defined by a point and a length to enforce perpendicular sides.
    """

    def __init__(self, center, side_length=None, radius=None):

        if radius is not None:
            # side_length = np.sqrt(2) * radius
            side_length = 2 * radius

        side_lengths = [side_length] * 2

        super(Square, self).__init__(center, side_lengths)


class Mesh(Entity):
    """A collection of Entities

    Attributes
    ----------
    faces : :py:obj:`list`
        A list of the Entities
    area : float
        The total area of the Entities
    population : int
        The number entities in the Mesh
    radius : float
        The radius of a bounding circle

    """

    def __init__(self, obj=None, faces=[]):
        self.faces = []
        self.area = 0
        self.population = 0
        self.radius = 0
        self._dim = 2

        if obj is not None:
            assert not faces
            self.import_triangle(obj)
        else:
            assert obj is None
            for face in faces:
                self.append(face)

    def __str__(self):
        return "Mesh(" + str(self.center) + ")"

    def __repr__(self):
        return "Mesh(faces={})".format(repr(self.faces))

    def import_triangle(self, obj):
        """Loads mesh data from a Python Triangle dict.
        """
        for face in obj['triangles']:
            p0 = Point(obj['vertices'][face[0], 0], obj['vertices'][face[0], 1])
            p1 = Point(obj['vertices'][face[1], 0], obj['vertices'][face[1], 1])
            p2 = Point(obj['vertices'][face[2], 0], obj['vertices'][face[2], 1])
            t = Triangle(p0, p1, p2)
            self.append(t)

    @property
    def center(self):
        center = Point([0, 0])
        if self.area > 0:
            for f in self.faces:
                center += f.center * f.area
            center /= self.area
        return center

    @property
    def bounding_box(self):
        """Return the axis-aligned bounding box as two numpy vectors."""
        xmin = np.full(self.dim, np.nan)
        xmax = np.full(self.dim, np.nan)

        for f in self.faces:
            fmin, fmax = f.bounding_box
            with np.errstate(invalid='ignore'):
                xmin = np.fmin(xmin, fmin)
                xmax = np.fmax(xmax, fmax)

        return xmin, xmax

    def append(self, t):
        """Add a triangle to the mesh."""
        self.population += 1
        # self.center = ((self.center * self.area + t.center * t.area) /
        #                (self.area + t.area))
        self.area += t.area

        if isinstance(t, Polygon):
            for v in t.vertices:
                self.radius = max(self.radius, self.center.distance(v))
        else:
            self.radius = max(
                self.radius,
                self.center.distance(t.center) + t.radius
            )

        self.faces.append(t)

    def pop(self, i=-1):
        """Pop i-th triangle from the mesh."""
        self.population -= 1
        self.area -= self.faces[i].area
        try:
            del self.__dict__['center']
        except KeyError:
            pass
        return self.faces.pop(i)

    def translate(self, vector):
        """Translate entity."""
        for t in self.faces:
            t.translate(vector)

    def rotate(self, theta, point=None, axis=None):
        """Rotate entity around an axis which passes through a point by theta
        radians."""
        for t in self.faces:
            t.rotate(theta, point, axis)

    def scale(self, vector):
        """Scale entity."""
        for t in self.faces:
            t.scale(vector)

    def contains(self, other):
        """Return whether this Mesh contains other.

        FOR ALL `x`,
        THERE EXISTS a face of the Mesh that contains `x`
        AND (ALL cut outs that contain `x` or THERE DOES NOT EXIST a cut out).
        """
        if isinstance(other, Point):
            x = other._x
        elif isinstance(other, np.ndarray):
            x = other
        elif isinstance(other, Polygon):
            x = _points_to_array(other.vertices)
            return np.all(self.contains(x))
        elif isinstance(other, Circle):
            warnings.warn("Didn't check that Mesh contains Circle.")
            return True
        else:
            raise NotImplementedError("Mesh.contains({})".format(type(other)))

        x = np.atleast_2d(x)

        # keep track of whether each point is contained in a face
        x_in_face = np.zeros(x.shape[0], dtype=bool)
        x_in_cut = np.zeros(x.shape[0], dtype=bool)
        has_cuts = False

        for f in self.faces:
            if f.sign < 0:
                has_cuts = True
                x_in_cut = np.logical_or(x_in_cut, f.contains(x))
            else:
                x_in_face = np.logical_or(x_in_face, f.contains(x))

        if has_cuts:
            return np.logical_and(x_in_face, x_in_cut)
        else:
            return x_in_face

    @property
    def patch(self):
        patches = []
        for f in self.faces:
            patches.append(f.patch)
        return patches
