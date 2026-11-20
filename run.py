"""Run from a source checkout without modifying the base environment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from human3d.cli import main

raise SystemExit(main())
