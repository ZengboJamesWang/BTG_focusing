"""Plotting helpers for the BTG focus-shift simulation.

All ray-diagram axes use z (optical axis) horizontally and x
(transverse coordinate) vertically.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from analysis import ray_x


def plot_focus_vs_thickness(thickness, vs_cover, vs_centre, vs_top):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(thickness, vs_cover, "o-", label="from cover-glass top surface")
    ax.plot(thickness, vs_centre, "s-", label="from particle centre")
    ax.plot(thickness, vs_top, "^-", label="from particle top surface")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("UV glue thickness")
    ax.set_ylabel("Focus position")
    ax.set_title("Focus position vs UV glue thickness")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_enhancement_vs_thickness(thickness, enhancement):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(thickness, enhancement, "o-", color="darkorange")
    ax.set_xlabel("UV glue thickness")
    ax.set_ylabel("Relative enhancement factor")
    ax.set_title("Relative enhancement factor vs UV glue thickness")
    fig.tight_layout()
    return fig


def plot_ray_diagram(ax, rays, radius, glue_thickness, focus_z, title):
    z_bottom = -0.3 * radius
    z_top = max(2.8 * radius, 1.1 * glue_thickness, min(focus_z + 0.8 * radius, 3.4 * radius))
    x_extent = 1.2 * radius

    # Cover glass (z < 0)
    ax.add_patch(
        patches.Rectangle(
            (z_bottom, -x_extent), -z_bottom, 2 * x_extent,
            color="lightblue", alpha=0.4, zorder=0, label="cover glass",
        )
    )

    # UV glue layer (0 <= z <= glue_thickness)
    if glue_thickness > 0:
        ax.add_patch(
            patches.Rectangle(
                (0, -x_extent), glue_thickness, 2 * x_extent,
                color="khaki", alpha=0.5, zorder=1, label="UV glue",
            )
        )

    # BTG sphere
    ax.add_patch(
        patches.Circle((radius, 0), radius, facecolor="white",
                        edgecolor="black", lw=1.5, zorder=2)
    )

    # Rays: red outside the BTG sphere, blue for the straight chord inside it.
    for r in rays:
        ax.plot(
            [z_bottom, r.points[0][1]],
            [r.points[0][0], r.points[0][0]],
            color="#e64b4b",
            lw=0.6,
            zorder=3,
        )
        if len(r.points) > 1:
            ax.plot(
                [r.points[0][1], r.points[1][1]],
                [r.points[0][0], r.points[1][0]],
                color="#1565c0",
                lw=0.8,
                zorder=3,
            )
            ax.plot(
                [r.points[0][1], r.points[1][1]],
                [r.points[0][0], r.points[1][0]],
                "o",
                color="#1565c0",
                ms=2.2,
                zorder=4,
            )
        if r.ok and r.direction[1] > 0:
            zs = [r.points[1][1]]
            xs = [r.points[1][0]]
            if len(r.points) > 2:
                zs.append(r.points[2][1])
                xs.append(r.points[2][0])
            zs.append(z_top)
            xs.append(ray_x(r, z_top))
            ax.plot(zs, xs, color="#e64b4b", lw=0.6, zorder=3)

    ax.axvline(focus_z, color="green", ls="--", lw=1,
               label=f"crossing focus z = {focus_z:.2f}")
    ax.set_xlim(z_bottom, z_top)
    ax.set_ylim(-x_extent, x_extent)
    ax.set_xlabel("z (optical axis)")
    ax.set_ylabel("x")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
