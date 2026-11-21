"""Archive-only tests; no MediaPipe, network or real model files are needed."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module():
    from scripts import vendor_mediapipe

    return vendor_mediapipe


def fixture(folder, tag="cp311-cp311-win_amd64"):
    m = module()
    root = Path(folder)
    resources = [
        "mediapipe/modules/pose_landmark/pose_landmark_full.tflite",
        "mediapipe/modules/pose_detection/pose_detection.tflite",
        "mediapipe/modules/face_landmark/face_landmark.tflite",
        "mediapipe/modules/face_detection/face_detection_short_range.tflite",
        "mediapipe/modules/pose_landmark/pose_landmark_cpu.binarypb",
        "mediapipe/modules/face_landmark/face_landmark_front_cpu.binarypb",
    ]
    expected = {"version": m.VERSION, "resources": []}
    wheel = root / "source" / m.WHEEL_NAME
    wheel.parent.mkdir(parents=True)
    with zipfile.ZipFile(wheel, "w") as archive:
        for i, name in enumerate(resources):
            raw = (f"synthetic-resource-{i}" * 3).encode()
            archive.writestr(name, raw)
            expected["resources"].append(
                {
                    "package_path": name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        prefix = f"mediapipe-{m.VERSION}.dist-info/"
        archive.writestr(
            prefix + "METADATA", f"Name: mediapipe\r\nVersion: {m.VERSION}\r\n"
        )
        archive.writestr(prefix + "WHEEL", f"Wheel-Version: 1.0\nTag: {tag}\n")
        archive.writestr(
            prefix + "LICENSE", "Apache License\nVersion 2.0\nSynthetic test fixture\n"
        )
    (root / m.VENDOR).mkdir(parents=True)
    (root / m.VENDOR / "expected_resources.json").write_text(json.dumps(expected))
    return wheel, expected


class MediaPipeVendorTests(unittest.TestCase):
    def test_git_preserves_original_metadata_line_endings(self):
        m = module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wheel, expected = fixture(folder)
            m.vendor_wheel(wheel, root, expected)
            raw = (root / m.VENDOR / "METADATA").read_bytes()
            self.assertIn(b"\r\n", raw)
            (root / ".gitattributes").write_bytes(
                (ROOT / ".gitattributes").read_bytes()
            )
            subprocess.run(
                ["git", "init", "-q", str(root)], check=True, capture_output=True
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "core.autocrlf=true",
                    "add",
                    ".gitattributes",
                    str(m.VENDOR),
                ],
                check=True,
                capture_output=True,
            )
            stored = subprocess.run(
                ["git", "-C", str(root), "show", ":third_party/mediapipe/METADATA"],
                check=True,
                capture_output=True,
            ).stdout
            self.assertEqual(stored, raw)

    def test_package_models_graphs_and_license_are_kept(self):
        m = module()
        with tempfile.TemporaryDirectory() as folder:
            wheel, expected = fixture(folder)
            result = m.vendor_wheel(wheel, Path(folder), expected)
            self.assertEqual(len(result["files"]), 10)
            self.assertTrue(m.verify(folder)["verified"])
            self.assertEqual(
                (Path(folder) / m.VENDOR / "wheels" / wheel.name).read_bytes(),
                wheel.read_bytes(),
            )
            self.assertIn(
                "Synthetic test fixture",
                (Path(folder) / m.VENDOR / "LICENSE").read_text(),
            )

    def test_resource_mismatch_stops_extraction(self):
        m = module()
        with tempfile.TemporaryDirectory() as folder:
            wheel, expected = fixture(folder)
            expected = copy.deepcopy(expected)
            expected["resources"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Resource differs"):
                m.vendor_wheel(wheel, Path(folder), expected)
            self.assertFalse((Path(folder) / m.VENDOR / "models").exists())

    def test_wrong_platform_is_rejected(self):
        m = module()
        with tempfile.TemporaryDirectory() as folder:
            wheel, expected = fixture(folder, tag="cp311-cp311-manylinux_x86_64")
            with self.assertRaisesRegex(ValueError, "Windows cp311"):
                m.vendor_wheel(wheel, Path(folder), expected)

    def test_corrupted_backup_is_detected(self):
        m = module()
        with tempfile.TemporaryDirectory() as folder:
            wheel, expected = fixture(folder)
            result = m.vendor_wheel(wheel, Path(folder), expected)
            (Path(folder) / result["files"][0]["path"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                m.verify(folder)

    def test_manifest_cannot_read_outside_vendor_directory(self):
        m = module()
        with tempfile.TemporaryDirectory() as folder:
            wheel, expected = fixture(folder)
            result = m.vendor_wheel(wheel, Path(folder), expected)
            result["files"][0]["path"] = "../../outside.dat"
            (Path(folder) / m.VENDOR / "manifest.json").write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                m.verify(folder)


if __name__ == "__main__":
    unittest.main()
