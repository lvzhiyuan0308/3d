"""Regression checks for coordinate, label and offline-runtime failures."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from human3d.settings import load_settings


class ConfigurationTests(unittest.TestCase):
    def test_paths_stay_relative_to_config(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.toml"
            config.write_text(
                '[paths]\nsmplx_model="models/female.npz"\noriginal_photo="photo.jpg"\nturnaround="views.png"\n',
                encoding="utf8",
            )
            settings = load_settings(config)
            self.assertEqual(settings.work, Path(folder).resolve() / "work")
            self.assertEqual(
                settings.model, Path(folder).resolve() / "models/female.npz"
            )

    def test_invalid_crop_fails_before_image_processing(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.toml"
            config.write_text(
                '[paths]\nsmplx_model="m"\noriginal_photo="p"\nturnaround="t"\n[[views]]\nname="front"\ncrop=[9,0,2,3]\nyaw_deg=0\n',
                encoding="utf8",
            )
            with self.assertRaisesRegex(ValueError, "positive area"):
                load_settings(config)


class GeometryTests(unittest.TestCase):
    def test_image_axis_and_occlusion(self):
        import numpy as np
        from human3d.core import project, rasterize

        uv, _ = project(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 0, 100, 50, 150)
        self.assertLess(uv[1, 1], uv[0, 1])
        xy = np.array([[1, 1], [6, 1], [1, 6], [1, 1], [6, 1], [1, 6]], dtype="float32")
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype="int32")
        ids, _, depth = rasterize(
            xy, np.array([0, 0, 0, 1, 1, 1], dtype="float32"), faces, 8, 8
        )
        self.assertEqual(ids[2, 2], 1)
        self.assertAlmostEqual(float(depth[2, 2]), 1.0)


class TextureTests(unittest.TestCase):
    def test_occluding_white_patch_cannot_leak_into_blue_source(self):
        import numpy as np
        from human3d.texture_semantics import masks_and_clean

        image = np.full((120, 100, 3), [140, 140, 140], dtype="uint8")
        labels = np.ones((120, 100), dtype="uint8")
        mask = np.ones_like(labels, dtype=bool)
        image[5:32, 12:88] = [230, 230, 230]
        labels[5:32, 12:88] = 2
        image[40:72, 12:88] = [180, 140, 100]
        labels[40:72, 12:88] = 3
        image[47:60, 40:52] = [245, 245, 245]
        labels[47:60, 40:52] = 2
        valid, clean = masks_and_clean(image, mask, labels)
        self.assertFalse(valid[3][52, 45])
        self.assertGreater(int(clean[3][52, 45, 0]) - int(clean[3][52, 45, 2]), 40)


class OfflineTests(unittest.TestCase):
    def test_socket_connect_rejected_before_transmission(self):
        script = "from human3d.offline import install_guard; import socket; install_guard(); socket.create_connection(('127.0.0.1',9))"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Local reconstruction forbids network activity", result.stderr)


if __name__ == "__main__":
    unittest.main()
