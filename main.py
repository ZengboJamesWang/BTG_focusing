"""BTG micro-sphere focus-shift simulation.

Estimates how the focal position and ray-concentration ("enhancement")
of a BTG micro-sphere lens change as a UV-glue coating gradually
embeds the sphere, using a fast approximate (geometric) ray-tracing
model -- no FDTD / FEM.

User inputs (see --help):
    --diameter      BTG particle diameter
    --wavelength    laser wavelength
    --max-thickness maximum UV glue thickness
    --steps         number of UV glue thickness steps

Everything else (refractive indices, ray aperture, ray count, plot
ranges) is fixed / derived automatically -- see the constants below.
"""

import argparse

import numpy as np

from analysis import enhancement_factor, find_crossing_focus
from ray_trace import trace_bundle, trace_symmetric_bundle

# ----------------------------------------------------------------------
# Fixed material parameters (edit here to change the optical system)
# ----------------------------------------------------------------------
N_SPHERE_EFF = 1.9  # effective refractive index used for the BTG sphere
N_BTG = N_SPHERE_EFF  # backwards-compatible alias
N_GLASS = 1.46  # cover glass (index-matched to the UV glue)
N_GLUE = 1.46   # UV glue
N_AIR = 1.0

# ----------------------------------------------------------------------
# Fixed numerical parameters
# ----------------------------------------------------------------------
N_RAYS_SWEEP = 300       # rays used for the focus / enhancement sweep
N_RAYS_DIAGRAM = 31      # rays used in the ray-diagram plots
INCIDENT_APERTURE_FRACTION = 1.0  # incident plane-wave aperture covers the projected particle diameter
FOCUS_APERTURE_FRACTION = INCIDENT_APERTURE_FRACTION
DIAGRAM_APERTURE_FRACTION = INCIDENT_APERTURE_FRACTION


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--diameter", type=float, default=20.0,
                   help="BTG particle diameter")
    p.add_argument("--wavelength", type=float, default=0.6,
                   help="laser wavelength (same length unit as diameter)")
    p.add_argument("--max-thickness", type=float, default=25.0,
                   help="maximum UV glue thickness (same length unit as diameter)")
    p.add_argument("--steps", type=int, default=11,
                   help="number of UV glue thickness steps")
    p.add_argument("--sphere-index", type=float, default=N_SPHERE_EFF,
                   help="effective refractive index used for the BTG sphere")
    return p.parse_args()


def sweep_thickness(
    radius,
    focus_aperture,
    thicknesses,
    wavelength,
    enhancement_aperture=None,
    n_sphere=N_SPHERE_EFF,
):
    """Compute focus height and relative enhancement for each thickness."""
    if enhancement_aperture is None:
        enhancement_aperture = focus_aperture

    focus_z = np.zeros_like(thicknesses)
    near_axis = np.zeros_like(thicknesses)

    for i, t in enumerate(thicknesses):
        focus_rays = trace_bundle(focus_aperture, N_RAYS_SWEEP, radius, t, n_sphere, N_GLUE, N_AIR)
        focus_z[i], near_axis[i], _, _ = find_crossing_focus(focus_rays, radius, t, wavelength)

    enh = np.array([enhancement_factor(enhancement_aperture, na, wavelength) for na in near_axis])
    enh_relative = enh / enh[0]
    return focus_z, enh_relative


def print_summary(thicknesses, focus_vs_cover, focus_vs_centre, focus_vs_top, enh_relative):
    header = f"{'glue t':>10} {'focus(cover)':>13} {'focus(centre)':>14} {'focus(top)':>11} {'rel. enh.':>10}"
    print(header)
    print("-" * len(header))
    for t, fc, fce, ft, e in zip(thicknesses, focus_vs_cover, focus_vs_centre, focus_vs_top, enh_relative):
        print(f"{t:10.3f} {fc:13.3f} {fce:14.3f} {ft:11.3f} {e:10.3f}")


def main():
    import matplotlib.pyplot as plt

    from plotting import plot_enhancement_vs_thickness, plot_focus_vs_thickness, plot_ray_diagram

    args = parse_args()

    radius = args.diameter / 2.0
    incident_aperture = INCIDENT_APERTURE_FRACTION * radius
    thicknesses = np.linspace(0.0, args.max_thickness, args.steps)

    focus_z, enh_relative = sweep_thickness(
        radius,
        incident_aperture,
        thicknesses,
        args.wavelength,
        enhancement_aperture=incident_aperture,
        n_sphere=args.sphere_index,
    )

    # Focus position reported relative to three reference planes
    focus_vs_cover = focus_z              # relative to cover-glass top surface (z = 0)
    focus_vs_centre = focus_z - radius    # relative to particle centre (z = R)
    focus_vs_top = focus_z - 2.0 * radius  # relative to particle top surface (z = 2R)

    print_summary(thicknesses, focus_vs_cover, focus_vs_centre, focus_vs_top, enh_relative)

    fig1 = plot_focus_vs_thickness(thicknesses, focus_vs_cover, focus_vs_centre, focus_vs_top)
    fig2 = plot_enhancement_vs_thickness(thicknesses, enh_relative)
    fig1.savefig("focus_vs_thickness.png", dpi=150)
    fig2.savefig("enhancement_vs_thickness.png", dpi=150)

    # Ray diagrams for the three canonical cases
    cases = {
        "No glue coating (t = 0)": 0.0,
        "Half-embedded (t = R)": radius,
        "Fully embedded (t = 2R)": 2.0 * radius,
    }
    fig3, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (title, t) in zip(axes, cases.items()):
        rays = trace_symmetric_bundle(incident_aperture, N_RAYS_DIAGRAM, radius, t, args.sphere_index, N_GLUE, N_AIR)
        f_z, _, _, _ = find_crossing_focus(rays, radius, t, args.wavelength)
        plot_ray_diagram(ax, rays, radius, t, f_z, title)
    fig3.tight_layout()
    fig3.savefig("ray_diagrams.png", dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
