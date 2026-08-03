"""Shared run-tracking utilities for the iNDI segmentation pipeline.

One run == one directory under an output root:

    <output-root>/run_<runID>/
        run_metadata.json     # list of per-stage records (machine-readable)
        run.log               # human-readable, appended by each stage
        <stage output dirs>/  # image_metadata/, nuclei_features/, ...

The metadata script (0_) mints the run ID with `new_run_id()` and creates the
run directory. Every downstream stage receives the run ID (via --run-id),
resolves the same directory, and appends its own record + log lines.

Stdlib only, so this imports cleanly from any script regardless of the conda
environment's third-party packages.
"""

from __future__ import annotations

import json
import os
import platform
import random
import socket
import string
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- Naming ----------------------------------------------------------------

RUN_PREFIX = "run_"
METADATA_FILENAME = "run_metadata.json"
LOG_FILENAME = "run.log"

_SUFFIX_ALPHABET = string.ascii_lowercase + string.digits


def new_run_id(suffix_len: int = 4) -> str:
    """Mint a fresh run ID: '<YYYYMMDD>_<HHMMSS>_<suffix>'.

    The timestamp makes runs sort chronologically and stay human-legible; the
    random suffix guarantees uniqueness even for two runs in the same second.
    Git state is recorded *inside* the metadata file, not in the ID.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "".join(random.choices(_SUFFIX_ALPHABET, k=suffix_len))
    return f"{stamp}_{suffix}"


def run_dirname(run_id: str) -> str:
    """Directory name for a run ID (adds the 'run_' prefix if not present)."""
    return run_id if run_id.startswith(RUN_PREFIX) else f"{RUN_PREFIX}{run_id}"


# --- Run directory resolution ----------------------------------------------

def create_run_dir(output_root: Path, run_id: str) -> Path:
    """Create and return <output-root>/run_<runID>/ (parents included).

    Used by the metadata stage (0_) at the start of a run.
    """
    run_dir = Path(output_root) / run_dirname(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_run_dir(output_root: Path, run_id: str, must_exist: bool = True) -> Path:
    """Return <output-root>/run_<runID>/ for a downstream stage.

    Raises a clear error if the run directory doesn't exist, so that running a
    downstream stage with a wrong/missing run ID fails loudly instead of
    silently creating a stray directory.
    """
    run_dir = Path(output_root) / run_dirname(run_id)
    if must_exist and not run_dir.is_dir():
        raise SystemExit(
            f"[error] run directory not found: {run_dir}\n"
            f"        (was the metadata stage run with --run-id {run_id} "
            f"and --output-root {output_root}?)"
        )
    return run_dir


def stage_dir(run_dir: Path, name: str) -> Path:
    """Create and return a named sub-directory inside a run dir.

    e.g. stage_dir(run_dir, 'nuclei_features') -> run_dir/nuclei_features/
    """
    d = Path(run_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- Git state -------------------------------------------------------------

def _git(*args: str) -> str | None:
    """Run a git command from the repo containing this file; None on failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_state() -> dict:
    """Capture git provenance for the run: commit, branch, dirty flag.

    'dirty' means there are uncommitted changes (tracked files modified or
    staged) at run time. When dirty, the exact commit no longer fully
    describes the code, so we also record the shortstat summary.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}

    status = _git("status", "--porcelain")
    dirty = bool(status)
    state = {
        "available": True,
        "commit": commit,
        "commit_short": commit[:12],
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
    }
    if dirty:
        # A compact summary of what's uncommitted, so a dirty run is still
        # partially reproducible / auditable.
        state["dirty_summary"] = _git("diff", "--shortstat") or ""
        state["dirty_files"] = [
            line[3:] for line in (status or "").splitlines()
        ]
    return state


# --- Environment snapshot --------------------------------------------------

def env_state() -> dict:
    """Capture a light environment snapshot for provenance."""
    return {
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cwd": os.getcwd(),
    }


# --- Metadata + log I/O ----------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _metadata_path(run_dir: Path) -> Path:
    return Path(run_dir) / METADATA_FILENAME


def _log_path(run_dir: Path) -> Path:
    return Path(run_dir) / LOG_FILENAME


def init_run_metadata(run_dir: Path, run_id: str) -> None:
    """Initialize run_metadata.json if it doesn't already exist.

    Idempotent: safe to call at the start of every stage. The top-level record
    holds run-wide fields; per-stage records are appended to 'stages'.
    """
    path = _metadata_path(run_dir)
    if path.exists():
        return
    doc = {
        "run_id": run_id,
        "created": _now_iso(),
        "git": git_state(),
        "environment": env_state(),
        "stages": [],
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")


def append_stage_record(run_dir: Path, record: dict) -> None:
    """Append one stage's record to run_metadata.json.

    Reads, appends, rewrites. Pipeline stages run sequentially (not concurrent
    writers to the same file), so a read-modify-write is safe here.
    """
    path = _metadata_path(run_dir)
    if path.exists():
        doc = json.loads(path.read_text())
    else:
        # Defensive: if init wasn't called, still produce a valid document.
        doc = {"run_id": None, "created": _now_iso(),
               "git": git_state(), "environment": env_state(), "stages": []}
    doc.setdefault("stages", []).append(record)
    path.write_text(json.dumps(doc, indent=2) + "\n")


def log(run_dir: Path, message: str, stage: str | None = None,
        echo: bool = True) -> None:
    """Append a timestamped line to run.log (and optionally echo to stdout)."""
    prefix = f"[{_now_iso()}]"
    if stage:
        prefix += f" [{stage}]"
    line = f"{prefix} {message}"
    with _log_path(run_dir).open("a") as fh:
        fh.write(line + "\n")
    if echo:
        print(line)


# --- Stage helper ----------------------------------------------------------

class StageRecorder:
    """Convenience wrapper a stage uses to record itself.

    Usage in a script:

        rec = StageRecorder(run_dir, stage="nucleus_segmentation",
                            run_id=run_id, params=collect_params(),
                            inputs={"metadata_dir": str(in_dir)})
        rec.log("segmenting 1234 DAPI images...")
        ...
        rec.finish(outputs={"features_dir": str(out_dir)},
                   summary={"total_nuclei": 5678})

    On finish(), a full record (with start/end times and duration) is appended
    to run_metadata.json.
    """

    def __init__(self, run_dir: Path, stage: str, run_id: str,
                 params: dict | None = None, inputs: dict | None = None,
                 argv: list[str] | None = None):
        self.run_dir = Path(run_dir)
        self.stage = stage
        self.run_id = run_id
        self.params = params or {}
        self.inputs = inputs or {}
        self.argv = argv if argv is not None else list(sys.argv)
        self._start = datetime.now(timezone.utc)
        init_run_metadata(self.run_dir, run_id)
        self.log(f"stage '{stage}' started")

    def log(self, message: str, echo: bool = True) -> None:
        log(self.run_dir, message, stage=self.stage, echo=echo)

    def finish(self, outputs: dict | None = None,
               summary: dict | None = None, status: str = "ok") -> dict:
        end = datetime.now(timezone.utc)
        record = {
            "stage": self.stage,
            "status": status,
            "started": self._start.astimezone().isoformat(timespec="seconds"),
            "ended": end.astimezone().isoformat(timespec="seconds"),
            "duration_s": round((end - self._start).total_seconds(), 1),
            "command": " ".join(self.argv),
            "params": self.params,
            "inputs": self.inputs,
            "outputs": outputs or {},
            "summary": summary or {},
        }
        append_stage_record(self.run_dir, record)
        self.log(f"stage '{self.stage}' finished ({record['duration_s']}s, "
                 f"status={status})")
        return record


# --- Argparse helpers ------------------------------------------------------

def add_run_args(parser, *, mints_run_id: bool) -> None:
    """Add the shared --output-root / --run-id arguments to a parser.

    mints_run_id=True  -> for stage 0_: --run-id is optional (auto-minted).
    mints_run_id=False -> for downstream stages: --run-id is required.
    """
    parser.add_argument(
        "--output-root", type=Path, default=Path("./outputs"),
        help="Root directory holding all run_<runID> directories "
             "(default: ./outputs).",
    )
    if mints_run_id:
        parser.add_argument(
            "--run-id", default=None,
            help="Run ID to use. If omitted, a fresh one is minted.",
        )
    else:
        parser.add_argument(
            "--run-id", required=True,
            help="Run ID minted by the metadata stage (0_). Required so this "
                 "stage reads from and writes to the correct run directory.",
        )