"""Run the source-checkout tests with an optional existing-environment adapter."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from human3d.offline import install_guard

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--legacy-user-site", action="store_true")
parser.add_argument(
    "--slow", action="store_true", help="Include two synthetic CPU fitting runs"
)
parser.add_argument("--report", default=str(ROOT / "work/test_results.json"))
args = parser.parse_args()
install_guard()
if args.legacy_user_site:
    import numpy  # noqa: F401 -- preload the base NumPy before user-site dependencies
    import site

    sys.path.insert(0, site.getusersitepackages())
    os.environ["HUMAN3D_LEGACY_USER_SITE"] = "1"
if args.slow:
    os.environ["HUMAN3D_SLOW_TESTS"] = "1"

import unittest

started = time.time()
suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = Path(args.report)
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(
    json.dumps(
        {
            "tests": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "passed": result.wasSuccessful(),
            "slow_enabled": os.environ.get("HUMAN3D_SLOW_TESTS") == "1",
            "seconds": time.time() - started,
        },
        indent=2,
    )
    + "\n",
    encoding="utf8",
)
raise SystemExit(not result.wasSuccessful())
