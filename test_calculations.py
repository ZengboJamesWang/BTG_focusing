import math
import unittest

from analysis import find_crossing_focus, find_external_focus, find_focus
from main import INCIDENT_APERTURE_FRACTION, N_SPHERE_EFF, sweep_thickness
from ray_trace import trace_bundle, trace_ray


N_BTG = N_SPHERE_EFF
N_GLUE = 1.46
N_AIR = 1.0
WAVELENGTH = 0.6


class CalculationTests(unittest.TestCase):
    def test_axial_ray_stays_on_axis(self):
        ray = trace_ray(0.0, 10.0, 20.0, N_BTG, N_GLUE, N_AIR)

        self.assertTrue(ray.ok)
        self.assertTrue(all(math.isclose(point[0], 0.0, abs_tol=1e-12) for point in ray.points))
        self.assertTrue(math.isclose(ray.direction[0], 0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(ray.direction[1], 1.0, abs_tol=1e-12))

    def test_paraxial_no_glue_focus_matches_ball_lens_limit(self):
        radius = 10.0
        aperture = 0.05 * radius
        rays = trace_bundle(aperture, 200, radius, 0.0, N_BTG, N_GLUE, N_AIR)

        focus_z, near_axis_fraction, spot_radius, transmission = find_focus(rays, radius, 0.0, WAVELENGTH)

        expected_focus_z = 2.0 * radius + (N_BTG * 2.0 * radius / (4.0 * (N_BTG - N_AIR)) - radius)
        self.assertAlmostEqual(focus_z, expected_focus_z, delta=0.05)
        self.assertLess(spot_radius, 0.002)
        self.assertAlmostEqual(near_axis_fraction, 1.0)
        self.assertAlmostEqual(transmission, 1.0)

    def test_transmission_is_area_weighted(self):
        radius = 10.0
        aperture = 0.9 * radius
        rays = trace_bundle(aperture, 200, radius, 10.0, N_BTG, N_GLUE, N_AIR)

        _, _, _, transmission = find_focus(rays, radius, 10.0, WAVELENGTH)
        count_fraction = sum(ray.ok for ray in rays) / len(rays)

        self.assertLess(transmission, count_fraction)
        self.assertAlmostEqual(transmission, sum(ray.weight for ray in rays if ray.ok))

    def test_encircled_waist_can_be_inside_particle_for_full_aperture(self):
        radius = 10.0
        aperture = INCIDENT_APERTURE_FRACTION * radius
        rays = trace_bundle(aperture, 200, radius, 0.0, N_BTG, N_GLUE, N_AIR)
        focus_z, _, _, _ = find_focus(rays, radius, 0.0, WAVELENGTH)
        external_focus_z, _, _, _ = find_external_focus(rays, radius, 0.0, WAVELENGTH)

        self.assertLess(focus_z, 2.0 * radius)
        self.assertGreaterEqual(external_focus_z, 2.0 * radius)

    def test_canonical_glue_cases_move_encircled_waist_outward(self):
        radius = 10.0
        aperture = INCIDENT_APERTURE_FRACTION * radius

        focus_positions = []
        for thickness in (0.0, radius, 2.0 * radius):
            rays = trace_bundle(aperture, 300, radius, thickness, N_BTG, N_GLUE, N_AIR)
            focus_z, _, _, _ = find_focus(rays, radius, thickness, WAVELENGTH)
            focus_positions.append(focus_z)

        no_glue, half_embedded, fully_embedded = focus_positions
        self.assertLess(no_glue, 2.0 * radius)
        self.assertGreater(half_embedded, 2.0 * radius)
        self.assertGreater(fully_embedded, half_embedded)

    def test_crossing_focus_moves_outward_with_embedding(self):
        radius = 10.0
        aperture = INCIDENT_APERTURE_FRACTION * radius

        focus_positions = []
        for thickness in (0.0, radius, 2.0 * radius):
            rays = trace_bundle(aperture, 300, radius, thickness, N_BTG, N_GLUE, N_AIR)
            focus_z, _, _, _ = find_crossing_focus(rays, radius, thickness, WAVELENGTH)
            focus_positions.append(focus_z)

        no_glue, half_embedded, fully_embedded = focus_positions
        self.assertGreaterEqual(no_glue, 2.0 * radius)
        self.assertGreater(half_embedded, no_glue)
        self.assertGreater(fully_embedded, half_embedded)


if __name__ == "__main__":
    unittest.main()
