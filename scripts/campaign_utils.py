from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CampaignRun:
    """A single benchmark run described by a Python module and its CLI arguments.

    The special token ``"{campaign}"`` inside *args* is replaced with the
    resolved campaign name at execution time.
    """

    name: str
    module: str
    args: tuple[str, ...]


def ensure_layout(area: str, campaign: str) -> tuple[Path, Path, Path]:
    """Create and return (log_dir, data_dir, analysis_dir) for the given area/campaign."""
    log_dir = Path("logs") / area / campaign
    data_dir = Path("data") / area / campaign
    analysis_dir = Path("data") / area / "analysis" / campaign
    log_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, data_dir, analysis_dir


def execute_runs(
    runs: list,
    campaign: str,
    log_dir: Path,
    analysis_dir: Path,
    execute: bool,
    header_lines: list[str],
    base_port: int | None = None,
) -> tuple[list[dict[str, object]], Path]:
    """Execute (or plan) all runs, write orchestrator.log and campaign_manifest.json.

    Args:
        runs: Objects with ``name``, ``module``, and ``args`` attributes.
        campaign: Resolved campaign name; replaces the ``"{campaign}"`` token in args.
        log_dir: Directory for per-run log files and orchestrator.log.
        analysis_dir: Directory for campaign_manifest.json.
        execute: When True, actually run each command; otherwise only plan.
        header_lines: Initial lines written to orchestrator.log before run entries.
        base_port: When set, appends ``--master-port <base_port + run_idx>`` to each
            command (needed for multi-process distributed runs).

    Returns:
        (manifest, manifest_path) where *manifest* is a list of per-run dicts.
    """
    manifest: list[dict[str, object]] = []
    orchestrator_lines = list(header_lines)

    for run_idx, run in enumerate(runs):
        rendered_args = tuple(campaign if arg == "{campaign}" else arg for arg in run.args)
        if base_port is not None:
            rendered_args = rendered_args + ("--master-port", str(base_port + run_idx))

        command = [sys.executable, "-m", run.module, *rendered_args]
        command_str = shlex.join(command)
        log_file = log_dir / f"{run.name}.log"
        entry: dict[str, object] = {
            "name": run.name,
            "command": command,
            "command_str": command_str,
            "log_file": str(log_file),
            "status": "planned",
        }
        orchestrator_lines.append(f"planned {run.name}: {command_str}")

        if execute:
            with log_file.open("w") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
            entry["return_code"] = completed.returncode
            entry["status"] = "completed" if completed.returncode == 0 else "failed"
            orchestrator_lines.append(f"finished {run.name}: return_code={completed.returncode}")

        manifest.append(entry)

    (log_dir / "orchestrator.log").write_text("\n".join(orchestrator_lines) + "\n")
    manifest_path = analysis_dir / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest, manifest_path
