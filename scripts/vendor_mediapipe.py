"""Back up the official Windows/Python 3.11 MediaPipe wheel and its resources.

Only Python standard-library modules are used. The wheel is inspected as data,
never installed or imported. Re-running with --verify requires no network.
"""

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from urllib.parse import urlparse
import zipfile

VERSION = "0.10.14"
WHEEL_NAME = f"mediapipe-{VERSION}-cp311-cp311-win_amd64.whl"
PYPI_URL = f"https://pypi.org/pypi/mediapipe/{VERSION}/json"
VENDOR = Path("third_party/mediapipe")
MAX_WHEEL_SIZE = 100 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_expected(root):
    path = root / VENDOR / "expected_resources.json"
    expected = json.loads(path.read_text(encoding="utf8"))
    if expected["version"] != VERSION or len(expected["resources"]) != 6:
        raise ValueError("Expected MediaPipe 0.10.14 with four models and two graphs")
    return expected


def download_wheel(root):
    with urllib.request.urlopen(PYPI_URL, timeout=60) as response:
        release = json.load(response)
    candidates = [item for item in release["urls"] if item["filename"] == WHEEL_NAME]
    if len(candidates) != 1:
        raise ValueError(f"Official PyPI release has no unique {WHEEL_NAME}")
    item = candidates[0]
    url = item["url"]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "files.pythonhosted.org":
        raise ValueError("Expected an official HTTPS PyPI file URL")
    if not 0 < item["size"] < MAX_WHEEL_SIZE:
        raise ValueError("Wheel exceeds the repository's ordinary-file size limit")
    target = root / VENDOR / "wheels" / WHEEL_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = item["digests"]["sha256"]
    if not target.is_file() or digest(target.read_bytes()) != expected_hash:
        partial = target.with_suffix(".whl.part")
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            partial.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        raw = partial.read_bytes()
        if len(raw) != item["size"] or digest(raw) != expected_hash:
            raise ValueError("Downloaded wheel differs from PyPI SHA-256/size")
        partial.replace(target)
    return target, {
        "release_url": PYPI_URL,
        "download_url": url,
        "sha256": expected_hash,
    }


def vendor_wheel(wheel, root, expected, provenance=None):
    root, wheel = Path(root).resolve(), Path(wheel).resolve()
    raw_wheel = wheel.read_bytes()
    if wheel.name != WHEEL_NAME or len(raw_wheel) >= MAX_WHEEL_SIZE:
        raise ValueError(f"Expected {WHEEL_NAME} below 100 MiB")
    provenance = provenance or {"source": "provided local wheel"}
    if provenance.get("sha256", digest(raw_wheel)) != digest(raw_wheel):
        raise ValueError("Wheel provenance SHA-256 does not match")
    prefix = f"mediapipe-{VERSION}.dist-info/"
    with zipfile.ZipFile(wheel) as archive:
        metadata = BytesParser().parsebytes(archive.read(prefix + "METADATA"))
        tags = archive.read(prefix + "WHEEL")
        if metadata["Name"].lower() != "mediapipe" or metadata["Version"] != VERSION:
            raise ValueError("Wrong MediaPipe package metadata")
        if b"Tag: cp311-cp311-win_amd64" not in tags:
            raise ValueError("Expected Windows cp311 wheel")
        payload = {}
        for resource in expected["resources"]:
            package_path = resource["package_path"]
            data = archive.read(package_path)
            if len(data) != resource["bytes"] or digest(data) != resource["sha256"]:
                raise ValueError(
                    "Resource differs from the verified local package: " + package_path
                )
            kind = "models" if package_path.endswith(".tflite") else "graphs"
            payload[f"{kind}/{Path(package_path).name}"] = (data, package_path)
        license_data = archive.read(prefix + "LICENSE")
        if b"Apache License" not in license_data or b"Version 2.0" not in license_data:
            raise ValueError("Expected the original Apache 2.0 license")
        for filename in ("LICENSE", "WHEEL", "METADATA"):
            payload[filename] = (archive.read(prefix + filename), prefix + filename)
        for name in archive.namelist():
            if name.startswith(prefix) and Path(name).name.upper().startswith("NOTICE"):
                payload[Path(name).name] = (archive.read(name), name)
    payload["wheels/" + WHEEL_NAME] = (raw_wheel, None)
    records = []
    for name, (data, package_path) in payload.items():
        target = root / VENDOR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append(
            {
                "path": target.relative_to(root).as_posix(),
                "bytes": len(data),
                "sha256": digest(data),
                "package_path": package_path,
            }
        )
    manifest = {
        "name": "mediapipe",
        "version": VERSION,
        "platform": "win_amd64",
        "python": "cp311",
        "provenance": provenance,
        "files": records,
    }
    (root / VENDOR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8"
    )
    return manifest


def verify(root):
    root = Path(root).resolve()
    vendor = (root / VENDOR).resolve()
    manifest = json.loads((vendor / "manifest.json").read_text(encoding="utf8"))
    if (manifest["version"], manifest["platform"], manifest["python"]) != (
        VERSION,
        "win_amd64",
        "cp311",
    ):
        raise ValueError("Unexpected vendor manifest version/platform")
    records = {}
    for record in manifest["files"]:
        path = (root / record["path"]).resolve()
        path.relative_to(vendor)
        data = path.read_bytes()
        if len(data) != record["bytes"] or digest(data) != record["sha256"]:
            raise ValueError("Vendored file checksum mismatch: " + record["path"])
        records[record.get("package_path")] = record
    for expected in load_expected(root)["resources"]:
        actual = records[expected["package_path"]]
        if (
            actual["sha256"] != expected["sha256"]
            or actual["bytes"] != expected["bytes"]
        ):
            raise ValueError("Resource fingerprint mismatch")
    required = {
        str(VENDOR / name).replace("\\", "/")
        for name in ("LICENSE", "WHEEL", "METADATA", "wheels/" + WHEEL_NAME)
    }
    if not required.issubset({record["path"] for record in manifest["files"]}):
        raise ValueError("Wheel or license metadata missing from manifest")
    return {
        "verified": True,
        "files": len(manifest["files"]),
        "bytes": sum(record["bytes"] for record in manifest["files"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--download", action="store_true")
    action.add_argument("--wheel", type=Path)
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.verify:
        expected = load_expected(root)
        wheel, source = download_wheel(root) if args.download else (args.wheel, None)
        vendor_wheel(wheel, root, expected, source)
    print(json.dumps(verify(root), indent=2))


if __name__ == "__main__":
    main()
