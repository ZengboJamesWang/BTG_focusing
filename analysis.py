"""Focus position and ray-concentration ("enhancement") analysis."""

import numpy as np


FOCUS_ENCIRCLED_FRACTION = 0.80
CROSSING_DENSITY_BANDWIDTH_FRACTION = 0.05


def ray_x(ray, z):
    """Transverse position of *ray* at height *z*.

    The ray path is piecewise-linear through its recorded vertices
    (points[0] = sphere entry point, ..., points[-1] = last
    refraction point). Below the first vertex the ray is still
    travelling along its incoming (vertical) path, so x is constant;
    between vertices x follows the straight chord; above the last
    vertex the ray follows its final direction -- unless it was lost
    to total internal reflection there, in which case NaN is returned.
    """
    pts = ray.points
    if z <= pts[0][1]:
        return pts[0][0]
    for i in range(len(pts) - 1):
        x0, z0 = pts[i]
        x1, z1 = pts[i + 1]
        if z <= z1:
            return x0 + (z - z0) * (x1 - x0) / (z1 - z0)
    if not ray.ok:
        return np.nan
    x0, z0 = pts[-1]
    dx, dz = ray.direction
    return x0 + (z - z0) * dx / dz


def _bundle_arrays(rays):
    """Return uniform NumPy arrays for a list of traced rays."""
    pts = [r.points + [r.points[-1]] * (3 - len(r.points)) for r in rays]
    return (
        np.array(pts, dtype=float),
        np.array([r.direction for r in rays], dtype=float),
        np.array([r.ok for r in rays], dtype=bool),
        np.array([getattr(r, "weight", 1.0) for r in rays], dtype=float),
    )


def _ray_x_grid(points, directions, z_grid):
    """Vectorised transverse ray positions for all z values."""
    x0, z0 = points[:, 0, 0], points[:, 0, 1]
    x1, z1 = points[:, 1, 0], points[:, 1, 1]
    x2, z2 = points[:, 2, 0], points[:, 2, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        slope01 = np.where(z1 != z0, (x1 - x0) / (z1 - z0), 0.0)
        slope12 = np.where(z2 != z1, (x2 - x1) / (z2 - z1), 0.0)
        slope_f = directions[:, 0] / directions[:, 1]

    z = z_grid[:, None]
    return np.where(
        z <= z0,
        x0,
        np.where(
            z <= z1,
            x0 + (z - z0) * slope01,
            np.where(
                z <= z2,
                x1 + (z - z1) * slope12,
                x2 + (z - z2) * slope_f,
            ),
        ),
    )


def _weighted_encircled_radius(abs_x, weights, fraction):
    """Radius enclosing *fraction* of weighted rays for each z row."""
    order = np.argsort(abs_x, axis=1)
    sorted_x = np.take_along_axis(abs_x, order, axis=1)
    sorted_w = weights[order]
    cumulative = np.cumsum(sorted_w, axis=1)
    targets = fraction * cumulative[:, -1]
    indices = (cumulative >= targets[:, None]).argmax(axis=1)
    return sorted_x[np.arange(abs_x.shape[0]), indices]


def _scan_bounds(points, directions, radius, glue_thickness, z_min):
    """Choose a finite z-range that covers the relevant caustic."""
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_f = directions[:, 0] / directions[:, 1]
        final_crossing = points[:, 2, 1] - points[:, 2, 0] / slope_f

    valid = np.isfinite(final_crossing) & (final_crossing > points[:, 2, 1])
    if valid.any():
        crossing_limit = np.nanpercentile(final_crossing[valid], 98.0) * 1.12
    else:
        crossing_limit = 0.0

    return max(
        z_min + 3.0 * radius,
        2.8 * radius,
        1.2 * glue_thickness,
        crossing_limit,
    )


def _axis_crossings(points, directions, weights, z_min):
    """Return weighted optical-axis crossing events for forward rays."""
    crossing_z = []
    crossing_weight = []
    for ray_points, direction, weight in zip(points, directions, weights):
        # The axial ray lies on x=0 continuously, so it is not a
        # discrete crossing event and would otherwise dominate the count.
        if abs(ray_points[0, 0]) < 1e-12:
            continue

        ray_crossings = []
        for p0, p1 in zip(ray_points[:-1], ray_points[1:]):
            x0, z0 = p0
            x1, z1 = p1
            if abs(x1 - x0) < 1e-12:
                continue
            u = -x0 / (x1 - x0)
            if -1e-12 <= u <= 1.0 + 1e-12:
                z_cross = z0 + u * (z1 - z0)
                if z_cross >= z_min - 1e-12:
                    ray_crossings.append(float(z_cross))

        x0, z0 = ray_points[-1]
        dx, dz = direction
        if abs(dx) > 1e-12:
            distance = -x0 / dx
            if distance >= -1e-12:
                z_cross = z0 + distance * dz
                if z_cross >= z_min - 1e-12:
                    ray_crossings.append(float(z_cross))

        # Avoid double-counting a crossing that lands exactly on an
        # interface vertex and is therefore seen by two adjacent segments.
        unique_crossings = []
        for z_cross in sorted(ray_crossings):
            if not unique_crossings or abs(z_cross - unique_crossings[-1]) > 1e-8:
                unique_crossings.append(z_cross)

        crossing_z.extend(unique_crossings)
        crossing_weight.extend([weight] * len(unique_crossings))

    return np.array(crossing_z, dtype=float), np.array(crossing_weight, dtype=float)


def find_crossing_focus(
    rays,
    radius,
    glue_thickness,
    wavelength,
    n_points=1600,
    z_min=None,
    density_bandwidth=None,
    encircled_fraction=FOCUS_ENCIRCLED_FRACTION,
):
    """Locate focus by the peak density of optical-axis crossings.

    This matches the intuitive ray-diagram definition: the focus is
    where the largest weighted number of off-axis rays cross x = 0.
    The central ray is excluded because it lies on the axis everywhere.
    A finite density bandwidth is required because sampled rays almost
    never cross at exactly identical z values.
    """
    points, directions, ok, weight = _bundle_arrays(rays)
    total_weight = float(np.sum(weight))

    forward = ok & (directions[:, 1] > 0)
    transmitted_weight = float(np.sum(weight[forward]))
    if transmitted_weight == 0.0:
        return np.nan, 0.0, np.nan, 0.0

    transmission = transmitted_weight / total_weight
    focus_points = points[forward]
    focus_directions = directions[forward]
    focus_weight = weight[forward]

    if z_min is None:
        z_min = 0.0
    if density_bandwidth is None:
        density_bandwidth = max(
            CROSSING_DENSITY_BANDWIDTH_FRACTION * radius,
            wavelength / 2.0,
        )

    crossing_z, crossing_weight = _axis_crossings(
        focus_points,
        focus_directions,
        focus_weight,
        z_min,
    )
    if crossing_z.size == 0:
        return find_focus(
            rays,
            radius,
            glue_thickness,
            wavelength,
            n_points=n_points,
            z_min=z_min,
            encircled_fraction=encircled_fraction,
        )

    if crossing_z.size == 1:
        focus_z = crossing_z[0]
    else:
        z_grid = np.linspace(crossing_z.min(), crossing_z.max(), n_points)
        scaled = (z_grid[None, :] - crossing_z[:, None]) / density_bandwidth
        density = np.sum(crossing_weight[:, None] * np.exp(-0.5 * scaled ** 2), axis=0)
        focus_z = z_grid[np.argmax(density)]

    abs_x = np.abs(_ray_x_grid(focus_points, focus_directions, np.array([focus_z])))[0]
    spot_radius = _weighted_encircled_radius(
        abs_x[None, :],
        focus_weight,
        encircled_fraction,
    )[0]
    near_axis_fraction = float(np.sum(focus_weight[abs_x < wavelength / 2.0]) / total_weight)
    return focus_z, near_axis_fraction, spot_radius, transmission


def find_external_crossing_focus(
    rays,
    radius,
    glue_thickness,
    wavelength,
    n_points=1600,
    density_bandwidth=None,
    encircled_fraction=FOCUS_ENCIRCLED_FRACTION,
):
    """Locate the outgoing-only optical-axis crossing-density focus."""
    transmitted = [r for r in rays if r.ok and r.direction[1] > 0]
    if not transmitted:
        return find_crossing_focus(
            rays,
            radius,
            glue_thickness,
            wavelength,
            n_points=n_points,
            density_bandwidth=density_bandwidth,
            encircled_fraction=encircled_fraction,
        )

    z_min = max(r.points[-1][1] for r in transmitted)
    return find_crossing_focus(
        rays,
        radius,
        glue_thickness,
        wavelength,
        n_points=n_points,
        z_min=z_min,
        density_bandwidth=density_bandwidth,
        encircled_fraction=encircled_fraction,
    )


def find_focus(
    rays,
    radius,
    glue_thickness,
    wavelength,
    n_points=1600,
    z_min=None,
    encircled_fraction=FOCUS_ENCIRCLED_FRACTION,
):
    """Locate the main best-focus waist of a transmitted ray bundle.

    Because a high-index microsphere has strong spherical aberration,
    the rays do not share one exact crossing point. Marginal rays can
    cross earlier than the bright central caustic. Instead of using the
    outermost ray envelope, this function finds the z plane where the
    radius enclosing ``encircled_fraction`` of transmitted ray power is
    smallest. That matches the visible main waist much better while
    still using the full incident aperture and area weights.

    By default the scan starts at the cover-glass top (z = 0), so the
    reported main focus can be inside the particle for t = 0.

    Returns (focus_z, near_axis_fraction, spot_radius, transmission):
      - near_axis_fraction: fraction of total incident power within
        wavelength/2 of the axis at focus_z (feeds enhancement_factor)
      - spot_radius: encircled ray radius at focus_z
      - transmission: incident-power fraction that survives without
        being lost to total internal reflection
    """
    points, directions, ok, weight = _bundle_arrays(rays)
    total_weight = float(np.sum(weight))

    forward = ok & (directions[:, 1] > 0)
    transmitted_weight = float(np.sum(weight[forward]))
    if transmitted_weight == 0.0:
        return np.nan, 0.0, np.nan, 0.0

    transmission = transmitted_weight / total_weight
    focus_points = points[forward]
    focus_directions = directions[forward]
    focus_weight = weight[forward]

    if z_min is None:
        z_min = 0.0
    z_max = _scan_bounds(focus_points, focus_directions, radius, glue_thickness, z_min)

    z_grid = np.linspace(z_min, z_max, n_points)
    abs_x = np.abs(_ray_x_grid(focus_points, focus_directions, z_grid))
    spot = _weighted_encircled_radius(abs_x, focus_weight, encircled_fraction)
    near_axis_fraction = ((abs_x < wavelength / 2.0) @ focus_weight) / total_weight
    idx = np.nanargmin(spot)
    return z_grid[idx], near_axis_fraction[idx], spot[idx], transmission


def find_external_focus(
    rays,
    radius,
    glue_thickness,
    wavelength,
    n_points=1600,
    encircled_fraction=FOCUS_ENCIRCLED_FRACTION,
):
    """Locate the outgoing focus after rays have left the interfaces.

    This uses the same encircled-power waist definition as
    ``find_focus`` but starts after the last recorded refraction point.
    """
    transmitted = [r for r in rays if r.ok and r.direction[1] > 0]
    if not transmitted:
        return find_focus(
            rays,
            radius,
            glue_thickness,
            wavelength,
            n_points=n_points,
            encircled_fraction=encircled_fraction,
        )

    z_min = max(r.points[-1][1] for r in transmitted)
    return find_focus(
        rays,
        radius,
        glue_thickness,
        wavelength,
        n_points=n_points,
        z_min=z_min,
        encircled_fraction=encircled_fraction,
    )


def enhancement_factor(aperture, near_axis_fraction, wavelength):
    """On-axis intensity enhancement relative to the incident beam.

    near_axis_fraction (from find_focus) is evaluated at the selected
    best-focus waist and gives the fraction of incident power within
    one diffraction-limited radius (wavelength/2) of the axis.
    """
    return near_axis_fraction * (aperture / wavelength) ** 2
