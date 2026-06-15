"""Streamlit web GUI for the BTG micro-sphere focus-shift simulation.

Run with:

    streamlit run app.py

The web app intentionally avoids Matplotlib so it starts quickly even on
systems where Matplotlib font discovery is slow or blocked.
"""

from html import escape
from urllib.parse import quote

import numpy as np
import streamlit as st

from analysis import (
    CROSSING_DENSITY_BANDWIDTH_FRACTION,
    FOCUS_ENCIRCLED_FRACTION,
    enhancement_factor,
    find_crossing_focus,
    find_external_crossing_focus,
    find_focus,
    ray_x,
)
from main import (
    INCIDENT_APERTURE_FRACTION,
    N_AIR,
    N_BTG,
    N_GLASS,
    N_GLUE,
    N_RAYS_DIAGRAM,
    N_RAYS_SWEEP,
    N_SPHERE_EFF,
    sweep_thickness,
)
from ray_trace import trace_bundle, trace_symmetric_bundle


def render_svg(svg, height):
    # Cross-browser compatible scaling (Fixes Chrome clipping issues)
    html_content = f"""
    <style>
        html, body {{ 
            margin: 0; 
            padding: 0; 
            height: 100%; 
            width: 100%; 
            overflow: hidden; 
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: transparent;
        }}
        svg {{ 
            max-width: 100%; 
            max-height: 100%; 
            width: auto; 
            height: auto;
            display: block;
        }}
    </style>
    {svg}
    """
    st.components.v1.html(html_content, height=height)


def svg_ray_diagram(rays, radius, glue_thickness, focus_z, title):
    z_bottom = -0.3 * radius
    z_top = max(2.8 * radius, 1.1 * glue_thickness, min(focus_z + 0.8 * radius, 3.4 * radius))
    x_extent = 1.2 * radius

    width = 560
    height = 380
    pad_l, pad_r, pad_t, pad_b = 54, 22, 30, 42
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    data_w = z_top - z_bottom
    data_h = 2 * x_extent
    scale = min(plot_w / data_w, plot_h / data_h)
    origin_x = pad_l + 0.5 * (plot_w - data_w * scale)
    origin_y = pad_t + 0.5 * (plot_h - data_h * scale)

    def sx(z):
        return origin_x + (z - z_bottom) * scale

    def sy(x):
        return origin_y + (x_extent - x) * scale

    def line(points, color, width_px=1.1, dash=""):
        coords = " ".join(f"{sx(z):.2f},{sy(x):.2f}" for z, x in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-width="{width_px}" stroke-linecap="round"{dash_attr}/>'
        )

    def dot(point, color, radius_px=2.4):
        z, x = point
        return f'<circle cx="{sx(z):.2f}" cy="{sy(x):.2f}" r="{radius_px}" fill="{color}"/>'

    items = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="18" text-anchor="middle" '
        f'font-size="14" font-family="sans-serif">{escape(title)}</text>',
        f'<rect x="{sx(z_bottom):.2f}" y="{sy(x_extent):.2f}" '
        f'width="{sx(0) - sx(z_bottom):.2f}" height="{plot_h:.2f}" '
        f'fill="#b7d6ef" opacity="0.45"/>',
    ]

    if glue_thickness > 0:
        items.append(
            f'<rect x="{sx(0):.2f}" y="{sy(x_extent):.2f}" '
            f'width="{sx(glue_thickness) - sx(0):.2f}" height="{plot_h:.2f}" '
            f'fill="#eadb75" opacity="0.55"/>'
        )

    items.append(
        f'<circle cx="{sx(radius):.2f}" cy="{sy(0):.2f}" '
        f'r="{radius * scale:.2f}" fill="#ffffff" '
        f'stroke="#111111" stroke-width="1.5"/>'
    )

    for ray in rays:
        entry = (ray.points[0][1], ray.points[0][0])
        exit_point = (ray.points[1][1], ray.points[1][0]) if len(ray.points) > 1 else entry
        items.append(line([(z_bottom, ray.points[0][0]), entry], "#e64b4b", 0.9))
        if len(ray.points) > 1:
            items.append(line([entry, exit_point], "#1565c0", 1.35))
            items.append(dot(entry, "#1565c0"))
            items.append(dot(exit_point, "#1565c0"))
        if ray.ok and ray.direction[1] > 0:
            after_exit = [exit_point]
            if len(ray.points) > 2:
                glue_air = (ray.points[2][1], ray.points[2][0])
                after_exit.append(glue_air)
                items.append(dot(glue_air, "#7a5c00"))
            after_exit.append((z_top, ray_x(ray, z_top)))
            items.append(line(after_exit, "#e64b4b", 0.9))

    items.append(line([(focus_z, -x_extent), (focus_z, x_extent)], "#1a8f3a", 1.4, "5 4"))
    items.append(line([(z_bottom, 0), (z_top, 0)], "#777777", 0.8, "3 5"))
    items.append(
        f'<text x="{sx(focus_z) + 5:.2f}" y="{pad_t + 16}" '
        f'font-size="11" font-family="sans-serif" fill="#1a8f3a">focus z={focus_z:.2f}</text>'
    )
    return f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" role="img">{"".join(items)}</svg>'


def svg_line_chart(x, series, title, y_label):
    width = 640
    height = 360
    pad_l, pad_r, pad_t, pad_b = 58, 22, 34, 44
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    x = np.asarray(x, dtype=float)
    all_y = np.concatenate([np.asarray(values, dtype=float) for _, values, _ in series])
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    if np.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    margin = 0.08 * (y_max - y_min)
    y_min -= margin
    y_max += margin

    def sx(value):
        return pad_l + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value):
        return pad_t + (y_max - value) / (y_max - y_min) * plot_h

    grid = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = pad_t + frac * plot_h
        value = y_max - frac * (y_max - y_min)
        grid.append(
            f'<line x1="{pad_l}" y1="{y:.2f}" x2="{width - pad_r}" y2="{y:.2f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="10" font-family="sans-serif" fill="#555">{value:.2g}</text>'
        )

    items = [
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" role="img">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="20" text-anchor="middle" '
        f'font-size="15" font-family="sans-serif">{escape(title)}</text>',
        *grid,
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" stroke="#333"/>',
        f'<line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#333"/>',
        f'<text x="{width / 2:.1f}" y="{height - 10}" text-anchor="middle" '
        f'font-size="11" font-family="sans-serif" fill="#444">UV glue thickness</text>',
        f'<text x="14" y="{height / 2:.1f}" transform="rotate(-90 14,{height / 2:.1f})" '
        f'text-anchor="middle" font-size="11" font-family="sans-serif" fill="#444">{escape(y_label)}</text>',
    ]

    legend_x = pad_l + 10
    for idx, (name, values, color) in enumerate(series):
        values = np.asarray(values, dtype=float)
        points = " ".join(f"{sx(xv):.2f},{sy(yv):.2f}" for xv, yv in zip(x, values))
        items.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for xv, yv in zip(x, values):
            items.append(f'<circle cx="{sx(xv):.2f}" cy="{sy(yv):.2f}" r="2.4" fill="{color}"/>')
        ly = pad_t + 18 + 18 * idx
        items.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 20}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        items.append(
            f'<text x="{legend_x + 26}" y="{ly + 4}" font-size="11" '
            f'font-family="sans-serif" fill="#333">{escape(name)}</text>'
        )

    items.append("</svg>")
    return "".join(items)


st.set_page_config(page_title="BTG Microsphere Focus Simulator", layout="wide")

st.title("BTG Microsphere Focus & Enhancement vs UV Glue Thickness")
st.caption(
    "Fast approximate geometric ray-tracing model, not FDTD/FEM. "
    f"Default effective sphere index n={N_SPHERE_EFF}; fixed indices: cover glass n={N_GLASS}, UV glue n={N_GLUE}, air n={N_AIR}."
)

with st.sidebar:
    st.header("Inputs")
    diameter = st.number_input("BTG particle diameter", min_value=0.1, value=20.0, step=1.0)
    wavelength = st.number_input("Laser wavelength", min_value=0.01, value=0.6, step=0.05)
    sphere_index = st.number_input(
        "Effective BTG sphere refractive index",
        min_value=1.0,
        value=float(N_BTG),
        step=0.01,
        format="%.3f",
    )
    max_thickness = st.number_input("Maximum UV glue thickness", min_value=0.0, value=25.0, step=1.0)
    steps = st.slider("Number of thickness steps", min_value=2, max_value=101, value=21)

radius = diameter / 2.0
incident_aperture = INCIDENT_APERTURE_FRACTION * radius
thicknesses = np.linspace(0.0, max_thickness, steps)

focus_z, enh_relative = sweep_thickness(
    radius,
    incident_aperture,
    thicknesses,
    wavelength,
    enhancement_aperture=incident_aperture,
    n_sphere=sphere_index,
)

focus_vs_cover = focus_z
focus_vs_centre = focus_z - radius
focus_vs_top = focus_z - 2.0 * radius

sweep_data = {
    "UV glue thickness": thicknesses,
    "Focus from cover": focus_vs_cover,
    "Focus from centre": focus_vs_centre,
    "Focus from top": focus_vs_top,
    "Relative enhancement": enh_relative,
}

col1, col2 = st.columns(2)
with col1:
    render_svg(
        svg_line_chart(
            thicknesses,
            [
                ("from cover", focus_vs_cover, "#1f77b4"),
                ("from centre", focus_vs_centre, "#2ca02c"),
                ("from top", focus_vs_top, "#d62728"),
            ],
            "Focus position vs UV glue thickness",
            "Focus position",
        ),
        height=390,
    )
with col2:
    render_svg(
        svg_line_chart(
            thicknesses,
            [("relative enhancement", enh_relative, "#d17a00")],
            "Relative enhancement vs UV glue thickness",
            "Relative enhancement",
        ),
        height=390,
    )

st.subheader("Ray-tracing diagrams: three canonical cases")
st.caption(
    "Green dashed line = crossing-density focus: the z position where the largest "
    "weighted number of off-axis rays cross the optical axis. "
    f"The density window is max({CROSSING_DENSITY_BANDWIDTH_FRACTION:.0%} of radius, wavelength/2)."
)
cases = {
    "No glue coating (t = 0)": 0.0,
    "Half-embedded (t = R)": radius,
    "Fully embedded (t = 2R)": 2.0 * radius,
}
cols = st.columns(3)
for col, (title, t) in zip(cols, cases.items()):
    rays = trace_symmetric_bundle(incident_aperture, N_RAYS_DIAGRAM, radius, t, sphere_index, N_GLUE, N_AIR)
    f_z, _, _, _ = find_crossing_focus(rays, radius, t, wavelength)
    with col:
        render_svg(svg_ray_diagram(rays, radius, t, f_z, title), height=400)

st.subheader("Custom glue thickness")
custom_t = st.slider(
    "UV glue thickness",
    min_value=0.0,
    max_value=max(max_thickness, 2.0 * radius),
    value=min(radius, max(max_thickness, 2.0 * radius)),
)
rows = trace_symmetric_bundle(incident_aperture, N_RAYS_DIAGRAM, radius, custom_t, sphere_index, N_GLUE, N_AIR)
focus_rays = trace_bundle(incident_aperture, N_RAYS_SWEEP, radius, custom_t, sphere_index, N_GLUE, N_AIR)
f_z, _, _, _ = find_crossing_focus(rows, radius, custom_t, wavelength)
dense_f_z, near_axis, spot, transmission = find_crossing_focus(focus_rays, radius, custom_t, wavelength)
external_f_z, _, external_spot, _ = find_external_crossing_focus(focus_rays, radius, custom_t, wavelength)
waist_f_z, _, waist_spot, _ = find_focus(focus_rays, radius, custom_t, wavelength)

left, right = st.columns([2, 1])
with left:
    render_svg(svg_ray_diagram(rows, radius, custom_t, f_z, f"t = {custom_t:.2f}"), height=420)
with right:
    st.metric("Displayed crossing focus, from cover", f"{f_z:.3f}")
    st.metric("Dense crossing focus, from cover", f"{dense_f_z:.3f}")
    st.metric("Dense crossing focus, from particle top", f"{dense_f_z - 2.0 * radius:.3f}")
    st.metric("Outgoing-only crossing focus", f"{external_f_z:.3f}")
    st.metric(f"{FOCUS_ENCIRCLED_FRACTION:.0%} waist focus, from cover", f"{waist_f_z:.3f}")
    st.metric("On-axis enhancement", f"{enhancement_factor(incident_aperture, near_axis, wavelength):.1f}×")
    st.metric(f"{FOCUS_ENCIRCLED_FRACTION:.0%} radius at crossing focus", f"{spot:.3f}")
    st.metric(f"{FOCUS_ENCIRCLED_FRACTION:.0%} waist radius", f"{waist_spot:.3f}")
    st.metric(f"Outgoing {FOCUS_ENCIRCLED_FRACTION:.0%} radius", f"{external_spot:.3f}")
    st.metric("Incident ray transmission", f"{transmission:.0%}")

st.subheader("Sweep data")
st.dataframe(sweep_data, width="stretch")
