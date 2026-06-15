# BTG Microsphere Focus Simulator

Fast approximate ray-tracing model for estimating how UV glue embedding changes
the focus position and qualitative enhancement of a BTG microsphere lens.

The model is intentionally lightweight: it uses geometric ray tracing rather
than FDTD or FEM, so it is useful for exploring trends quickly rather than for
predicting exact electromagnetic near-field enhancement.

## What It Simulates

- Plane-wave illumination from bottom to top.
- BTG microsphere sitting on top of a cover glass.
- UV glue progressively embedding the particle.
- Refraction through the sphere and glue/air interfaces.
- Focus position versus UV glue thickness.
- Relative enhancement estimated from ray concentration near focus.
- Ray-tracing diagrams for no glue, half-embedded, and fully embedded cases.

Default refractive indices:

| Material | Refractive index |
| --- | ---: |
| BTG sphere | 1.90 |
| Cover glass | 1.46 |
| UV glue | 1.46 |
| Air | 1.00 |

## Focus Definition

The green dashed focus marker in the web app uses a **crossing-density focus**:

1. Trace the transmitted off-axis rays.
2. Find where each ray crosses the optical axis.
3. Ignore the central ray because it lies on the optical axis everywhere.
4. Use a small smoothing window to find where the largest weighted number of
   rays cross the axis.

The app also reports an alternative **80% waist focus** for comparison. That
metric finds the z-plane where the radius enclosing 80% of transmitted ray
power is smallest.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Web App

```bash
streamlit run app.py
```

The app lets you adjust:

- BTG particle diameter
- Laser wavelength
- Effective BTG sphere refractive index
- Maximum UV glue thickness
- Number of thickness steps

## Run The Command-Line Simulation

```bash
python main.py
```

Example with custom parameters:

```bash
python main.py --diameter 20 --wavelength 0.6 --max-thickness 25 --steps 21
```

The command-line script prints a summary table and saves:

- `focus_vs_thickness.png`
- `enhancement_vs_thickness.png`
- `ray_diagrams.png`

## Run Tests

```bash
python -m unittest -v
```

## Project Layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit web app and SVG plotting |
| `main.py` | Command-line simulation entry point |
| `ray_trace.py` | Ray tracing through sphere/glue/air geometry |
| `optics.py` | Vector refraction and geometry helpers |
| `analysis.py` | Focus and enhancement calculations |
| `plotting.py` | Matplotlib plots for command-line output |
| `test_calculations.py` | Unit tests for optical calculations |
| `prompt` | Original project brief |

## Notes And Limitations

- This is a geometric-optics approximation.
- It does not include wave interference, diffraction, polarization, scattering,
  absorption, or near-field Maxwell effects.
- Enhancement is qualitative and based on ray concentration, not exact optical
  field intensity.
- The model is best used to compare trends as glue thickness changes.
