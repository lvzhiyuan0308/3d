"""Command line orchestration; heavy packages load only inside stage workers."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .settings import load_settings

STAGES = {
    "prepare": "prepare",
    "check-core": "check_core",
    "fit": "fit",
    "canonical": "update_canonical",
    "geometry": "geometry",
    "export": "export_local",
    "validate": "validate_delivery",
    "validate-revision": "validate_revision",
    "preview": "make_review",
}


def run_stage(stage, settings, legacy=False):
    env = os.environ.copy()
    env["HUMAN3D_CONFIG"] = str(settings.config)
    env["HUMAN3D_LEGACY_USER_SITE"] = "1" if legacy else "0"
    command = [sys.executable]
    if legacy:
        command.append("-s")
    command.extend([str(Path(__file__).with_name("_stage.py")), stage])
    print(f"Running: {stage}", flush=True)
    subprocess.run(command, env=env, check=True)


def bootstrap(settings, source, overwrite=False):
    """Import a previous local run, never model weights or authentication data."""
    source = Path(source).resolve()
    files = [
        (source / "inputs/asset_01.jpg", settings.photo),
        (source / "inputs/asset_05.png", settings.turnaround),
    ]
    for name in ("prepared", "fit"):
        files.extend(
            (p, settings.work / name / p.name)
            for p in (source / name).glob("*")
            if p.is_file()
        )
    cache = source / "surface/face_landmarks.json"
    if cache.is_file():
        files.append((cache, settings.work / "surface/face_landmarks.json"))
    for src, dst in files:
        if not src.is_file():
            raise FileNotFoundError(src)
        if dst.exists() and not overwrite:
            raise FileExistsError(
                f"Destination exists: {dst}. Use --overwrite only if intended."
            )
        if src == dst:
            raise ValueError("Bootstrap source and destination must differ")
    if not (source / "fit/fit_result.npz").is_file():
        raise FileNotFoundError("Previous run has no fit/fit_result.npz")
    for src, dst in files:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(
        json.dumps(
            {"copied_files": len(files), "work_dir": str(settings.work)},
            ensure_ascii=False,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Local multiview SMPL-X reconstruction"
    )
    parser.add_argument(
        "--config",
        default="config.local.toml",
        help="TOML config; all paths are relative to this file",
    )
    parser.add_argument(
        "--legacy-user-site",
        action="store_true",
        help="Use existing Anaconda NumPy before user-site GPU packages",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check dependencies, model and local inputs")
    run = commands.add_parser(
        "run", help="Resume or fit, construct, export and validate"
    )
    run.add_argument(
        "--refit", action="store_true", help="Recompute preparation and body fit"
    )
    stage = commands.add_parser("stage", help="Execute one stage")
    stage.add_argument("name", choices=STAGES)
    init = commands.add_parser(
        "bootstrap", help="Import this project's previous V2 local run"
    )
    init.add_argument("--source", required=True)
    init.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if args.command == "bootstrap":
            bootstrap(settings, args.source, args.overwrite)
            return 0
        settings.work.mkdir(parents=True, exist_ok=True)
        if args.command == "doctor":
            run_stage("doctor", settings, args.legacy_user_site)
        elif args.command == "stage":
            run_stage(STAGES[args.name], settings, args.legacy_user_site)
        else:
            stages = []
            if args.refit or not (settings.work / "fit/fit_result.npz").is_file():
                stages.extend(["prepare", "check_core", "fit"])
            stages.extend(
                [
                    "update_canonical",
                    "geometry",
                    "export_local",
                    "validate_delivery",
                    "validate_revision",
                    "make_review",
                ]
            )
            for name in stages:
                run_stage(name, settings, args.legacy_user_site)
            print(f"Outputs: {settings.work / 'delivery'}")
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"human3d: {exc}\n")
    return 0
