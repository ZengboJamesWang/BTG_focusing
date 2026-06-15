"""Ray tracing of a single ray through the BTG-microsphere / UV-glue /
air stack.

Geometry (2D meridional cross-section, axisymmetric about x = 0):

    z < 0                : cover glass (index-matched to the UV glue,
                            so it has no optical effect and is ignored)
    0 <= z <= glue_thick : UV glue, surrounds the sphere up to a flat
                            top surface at z = glue_thickness
    z > glue_thickness   : air
    sphere               : centred at (0, radius), occupies 0 <= z <= 2*radius

A ray enters travelling straight up (+z). It refracts on entering the
sphere, refracts again on leaving it, and -- if it leaves into the
glue below the glue's flat top surface -- refracts a third time at
that flat glue/air interface.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from optics import refract, sphere_normal, second_sphere_intersection


@dataclass
class Ray:
    """A traced ray.

    points    : vertices (x, z) from the sphere entry point onward
    direction : final unit propagation direction
    ok        : False if the ray was lost to total internal reflection
    weight    : fraction of incident circular aperture power represented
                by this sampled ray
    """

    points: List[np.ndarray] = field(default_factory=list)
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    ok: bool = True
    weight: float = 1.0


def trace_ray(h, radius, glue_thickness, n_btg, n_glue, n_air):
    """Trace one ray entering the sphere at transverse offset *h*."""
    center = np.array([0.0, radius])

    # --- entry point on the lower hemisphere -----------------------
    z1 = radius - np.sqrt(radius ** 2 - h ** 2)
    p1 = np.array([h, z1])
    n_below = n_glue if z1 <= glue_thickness else n_air

    d_in = refract(np.array([0.0, 1.0]), sphere_normal(p1, center), n_below, n_btg)
    if d_in is None:
        return Ray([p1], np.array([0.0, 1.0]), ok=False)

    # --- exit point on the upper hemisphere -------------------------
    p2 = second_sphere_intersection(p1, d_in, center)
    n_above = n_glue if p2[1] <= glue_thickness else n_air

    d_out = refract(d_in, sphere_normal(p2, center), n_btg, n_above)
    if d_out is None:
        return Ray([p1, p2], d_in, ok=False)

    points = [p1, p2]
    direction = d_out

    # --- optional third refraction at the flat glue/air surface -----
    if n_above == n_glue and p2[1] < glue_thickness and d_out[1] > 0:
        s = (glue_thickness - p2[1]) / d_out[1]
        p3 = p2 + s * d_out
        d_final = refract(d_out, np.array([0.0, 1.0]), n_glue, n_air)
        if d_final is None:
            return Ray(points + [p3], d_out, ok=False)
        points.append(p3)
        direction = d_final

    return Ray(points, direction, ok=True)


def trace_bundle(aperture, n_rays, radius, glue_thickness, n_btg, n_glue, n_air):
    """Trace a bundle of parallel rays spanning the aperture (h = 0..aperture)."""
    h_values = np.linspace(0.0, aperture, n_rays)
    rays = [
        trace_ray(h, radius, glue_thickness, n_btg, n_glue, n_air)
        for h in h_values
    ]

    # Axisymmetric plane-wave power is proportional to annular area, not
    # to radial sample count. Use midpoint annuli around each sampled ray.
    if n_rays > 1 and aperture > 0:
        edges = np.empty(n_rays + 1)
        edges[0] = 0.0
        edges[-1] = aperture
        edges[1:-1] = 0.5 * (h_values[:-1] + h_values[1:])
        weights = edges[1:] ** 2 - edges[:-1] ** 2
        weights = weights / weights.sum()
        for ray, weight in zip(rays, weights):
            ray.weight = float(weight)

    return rays


def trace_symmetric_bundle(aperture, n_rays, radius, glue_thickness, n_btg, n_glue, n_air):
    """Trace a diagram bundle spanning both sides of the optical axis."""
    h_values = np.linspace(-aperture, aperture, n_rays)
    return [
        trace_ray(h, radius, glue_thickness, n_btg, n_glue, n_air)
        for h in h_values
    ]
