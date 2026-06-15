"""Basic vector-optics primitives used by the ray tracer.

Everything here works in a 2D (x, z) meridional plane: z is the optical
axis (direction of plane-wave propagation, bottom -> top), x is the
transverse / radial coordinate.
"""

import numpy as np


def normalize(v):
    return v / np.linalg.norm(v)


def refract(direction, normal, n1, n2):
    """Refract a unit direction vector at an interface.

    direction : incident ray direction (unit vector)
    normal    : surface normal (unit vector, either orientation)
    n1, n2    : refractive indices of the incident and transmitted media

    Returns the refracted unit direction, or None on total internal
    reflection.
    """
    d = normalize(direction)
    n = normalize(normal)

    cosi = -np.dot(d, n)
    if cosi < 0:
        n = -n
        cosi = -cosi

    eta = n1 / n2
    sin2t = eta ** 2 * (1.0 - cosi ** 2)
    if sin2t > 1.0:
        return None  # total internal reflection

    cost = np.sqrt(1.0 - sin2t)
    return eta * d + (eta * cosi - cost) * n


def sphere_normal(point, center):
    """Outward unit normal of a sphere at a point on its surface."""
    return normalize(point - center)


def second_sphere_intersection(point, direction, center):
    """Other intersection of a line with a sphere.

    Given a point already on the sphere and a propagation direction,
    return the second point where the straight line meets the sphere.
    """
    d = normalize(direction)
    to_center = point - center
    s = -2.0 * np.dot(d, to_center)
    return point + s * d
