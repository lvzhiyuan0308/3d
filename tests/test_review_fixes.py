"""Regression tests for the review fixes: fit terms, config options, vectorised helpers."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from human3d.settings import load_settings

BASE = '[paths]\nsmplx_model="m"\noriginal_photo="p"\nturnaround="t"\n'


def settings_from(extra=""):
    with tempfile.TemporaryDirectory() as folder:
        config = Path(folder) / "config.toml"
        config.write_text(BASE + extra, encoding="utf8")
        return load_settings(config)


class SettingsTests(unittest.TestCase):
    def test_defaults_reproduce_original_behaviour(self):
        s = settings_from()
        self.assertEqual(s.height_m, 1.695)
        self.assertFalse(s.height_is_measured)
        self.assertFalse(s.fit_use_view_weights)
        self.assertEqual(s.fit_height_weight, 0.4)

    def test_subject_and_fit_options(self):
        s = settings_from(
            "[subject]\nheight_m=1.80\nheight_measured=true\n"
            "[fit]\nuse_view_weights=true\nheight_weight=2.5\n"
        )
        self.assertEqual((s.height_m, s.height_is_measured), (1.80, True))
        self.assertTrue(s.fit_use_view_weights)
        self.assertEqual(s.fit_height_weight, 2.5)

    def test_invalid_values_rejected(self):
        with self.assertRaisesRegex(ValueError, "height_m"):
            settings_from("[subject]\nheight_m=17\n")
        with self.assertRaisesRegex(ValueError, "height_weight"):
            settings_from("[fit]\nheight_weight=-1\n")


class FitTermTests(unittest.TestCase):
    def test_keypoint_term_is_resolution_independent(self):
        import torch

        from human3d.fit_terms import keypoint_term

        g = torch.Generator().manual_seed(0)
        pred = torch.rand(4, 15, 2, generator=g) * 500
        gt = pred + torch.randn(4, 15, 2, generator=g) * 12
        conf = torch.rand(4, 15, generator=g)
        heights = torch.full((4,), 534.0)
        base = keypoint_term(pred, gt, conf, heights)
        for k in (0.5, 2.0):  # same picture at another resolution
            scaled = keypoint_term(pred * k, gt * k, conf, heights * k)
            self.assertAlmostEqual(float(base), float(scaled), places=6)

    def test_keypoint_term_uses_each_views_own_height(self):
        import torch

        from human3d.fit_terms import keypoint_term

        pred = torch.zeros(2, 3, 2)
        gt = torch.full((2, 3, 2), 20.0)
        conf = torch.ones(2, 3)
        tall = keypoint_term(pred, gt, conf, torch.tensor([534.0, 534.0]))
        short = keypoint_term(pred, gt, conf, torch.tensor([200.0, 200.0]))
        mixed = keypoint_term(pred, gt, conf, torch.tensor([534.0, 200.0]))
        self.assertLess(float(tall), float(mixed))
        self.assertLess(float(mixed), float(short))
        self.assertAlmostEqual(float(mixed), (float(tall) + float(short)) / 2, places=6)

    def test_view_weights(self):
        import torch

        from human3d.fit_terms import view_weight_vector

        views = [{"name": "a", "geometry_weight": 0.6}, {"name": "b"}]
        self.assertTrue(torch.equal(view_weight_vector(views), torch.ones(2)))
        w = view_weight_vector(views, enabled=True)
        self.assertAlmostEqual(float(w[0]), 0.6, places=6)
        self.assertEqual(float(w[1]), 1.0)
        with self.assertRaises(ValueError):
            view_weight_vector([{"name": "x", "geometry_weight": 0}], enabled=True)


class Geom2DTests(unittest.TestCase):
    def test_matches_opencv_point_polygon_test(self):
        import cv2

        from human3d.geom2d import points_in_convex_polygon

        rng = np.random.default_rng(1)
        hull = cv2.convexHull((rng.random((40, 2)) * 200).astype("float32"))
        pts = rng.random((3000, 2)) * 240 - 20
        expected = np.array(
            [cv2.pointPolygonTest(hull, tuple(map(float, p)), False) >= 0 for p in pts]
        )
        np.testing.assert_array_equal(points_in_convex_polygon(pts, hull), expected)

    def test_boundary_and_orientation(self):
        from human3d.geom2d import points_in_convex_polygon

        square = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float)
        pts = np.array([[0, 0], [2, 0], [2, 2], [5, 2], [-0.1, 2]])
        expected = [True, True, True, False, False]
        self.assertEqual(points_in_convex_polygon(pts, square).tolist(), expected)
        self.assertEqual(points_in_convex_polygon(pts, square[::-1]).tolist(), expected)
        self.assertFalse(points_in_convex_polygon(pts, square[:2]).any())


class LayeredVectorisationTests(unittest.TestCase):
    def test_scalp_cap_mask_equals_reference_loop(self):
        from human3d.layered import scalp_cap_mask

        rng = np.random.default_rng(2)
        can = rng.normal(size=(500, 3)).astype("float32") * 0.1
        faces = rng.integers(0, 500, size=(3000, 3))
        in_head = rng.random(500) > 0.3
        hc = np.array([0.0, 0.05, 0.02], "float32")
        reference = []  # the original per-face loop
        for tri in faces:
            p = can[tri].mean(0)
            front = p[2] - hc[2]
            threshold = hc[1] - 0.055 + np.clip((front + 0.045) / 0.10, 0, 1) * 0.13
            reference.append(in_head[tri].all() and p[1] > threshold)
        got = scalp_cap_mask(can, faces, in_head, hc)
        np.testing.assert_array_equal(got, np.array(reference))
        self.assertTrue(0 < got.sum() < len(faces))

    def test_boundary_cap_faces_equals_reference_loop(self):
        from human3d.layered import boundary_cap_faces

        boundary = np.array([[0, 1], [1, 2], [2, 0], [5, 7]])
        n = 10
        reference = []
        for a, b in boundary:
            reference.extend([(b, a, a + n), (b, a + n, b + n)])
        np.testing.assert_array_equal(
            boundary_cap_faces(boundary, n), np.array(reference)
        )
        self.assertEqual(boundary_cap_faces(np.zeros((0, 2), int), n).shape, (0, 3))


@unittest.skipUnless(
    os.environ.get("HUMAN3D_SLOW_TESTS") == "1",
    "set HUMAN3D_SLOW_TESTS=1 (about 2.5 minutes)",
)
class FitLayoutInvarianceTest(unittest.TestCase):
    """The fit must not depend on the pixel size of the panels (a hard-coded 534 once broke this)."""

    def run_fit(self, folder, panel_h):
        from synthetic import make_inputs

        config = make_inputs(folder, panel_h=panel_h)
        env = dict(os.environ, HUMAN3D_CONFIG=str(config), HUMAN3D_LEGACY_USER_SITE="0")
        stage = ROOT / "src/human3d/_stage.py"
        subprocess.run(
            [sys.executable, str(stage), "fit"],
            env=env,
            check=True,
            capture_output=True,
        )
        return np.load(Path(folder) / "work/fit/fit_result.npz")["betas"]

    def test_betas_do_not_depend_on_panel_height(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            tall, short = self.run_fit(a, 534), self.run_fit(b, 300)
        self.assertLess(float(np.abs(tall - short).max()), 0.1)


if __name__ == "__main__":
    unittest.main()
