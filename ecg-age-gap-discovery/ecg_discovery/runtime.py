"""Determinism, device selection, and structured run logging.

Every entrypoint in this repository starts by calling :func:`set_global_seed`
and opening a :class:`RunContext`. Together they guarantee that a result in the
paper can be regenerated: the run directory records the seed, the resolved
config, the git commit (when available), the installed package versions, and a
JSON-lines stream of per-epoch metrics.

This module is a small addition to the repository layout given in the build
plan. It exists so that seeding and logging are implemented once rather than
copy-pasted into each of the five ``scripts/`` entrypoints.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["set_global_seed", "resolve_device", "RunContext", "git_commit_hash"]


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed every random number generator this project draws from.

    Parameters
    ----------
    seed:
        Non-negative integer seed. The same seed with the same config must
        reproduce the same numbers.
    deterministic_torch:
        If true, also disables cuDNN autotuning and requests deterministic
        algorithms. This costs some GPU throughput, which is the right trade for
        a project whose headline claim is a numerical decomposition others must
        be able to reproduce.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:  # signal-processing-only usage should not require torch
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> str:
    """Turn a config ``device`` string into a concrete torch device string.

    ``auto`` prefers CUDA, then Apple Silicon's MPS, then CPU. Everything in this
    repository is small enough to run on CPU, which is the intended development
    path.
    """
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("device='mps' requested but MPS is not available")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def git_commit_hash(repo_root: Path | None = None) -> str | None:
    """Return the current git commit hash, or ``None`` outside a git repository.

    Recorded in every run directory so a result can be tied to the exact code
    that produced it.
    """
    root = repo_root or Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _package_versions() -> dict[str, str]:
    """Version strings of the packages whose behaviour can change results."""
    versions: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("torch", "numpy", "scipy", "pandas", "sklearn", "wfdb"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


class RunContext:
    """A timestamped output directory with provenance and JSON-lines metrics.

    Creates ``<runs_dir>/<experiment_name>/<UTC timestamp>/`` containing:

    ``config.json``
        The exact resolved configuration, seed, git commit and package versions.
    ``metrics.jsonl``
        One JSON object per logged record (typically one per epoch).
    ``artifacts/``
        Checkpoints, plots and result tables written by the run.

    Use as a context manager::

        with RunContext("ecg_age_regressor", "runs", seed=0, configs={...}) as run:
            run.log({"epoch": 0, "train_loss": 12.3})
    """

    def __init__(
        self,
        experiment_name: str,
        runs_dir: str | Path = "runs",
        *,
        seed: int,
        configs: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = Path(runs_dir) / experiment_name / timestamp
        self.artifacts_dir = self.dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.seed = seed
        self._metrics_path = self.dir / "metrics.jsonl"
        self._metrics_handle = None

        payload: dict[str, Any] = {
            "experiment_name": experiment_name,
            "timestamp_utc": timestamp,
            "seed": seed,
            "git_commit": git_commit_hash(),
            "packages": _package_versions(),
            "argv": sys.argv,
            "configs": {
                key: (_as_plain(value)) for key, value in (configs or {}).items()
            },
        }
        if extra:
            payload["extra"] = _as_plain(extra)
        (self.dir / "config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        self.provenance = payload

    def __enter__(self) -> "RunContext":
        self._metrics_handle = self._metrics_path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._metrics_handle is not None:
            self._metrics_handle.close()
            self._metrics_handle = None

    def log(self, record: dict[str, Any]) -> None:
        """Append one JSON record to ``metrics.jsonl`` and flush it immediately.

        Flushing on every record means a killed run still leaves usable partial
        metrics behind.
        """
        line = json.dumps(_as_plain(record), sort_keys=True, default=str)
        if self._metrics_handle is None:
            with self._metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        else:
            self._metrics_handle.write(line + "\n")
            self._metrics_handle.flush()

    def save_json(self, name: str, payload: Any) -> Path:
        """Write a JSON artifact into the run's ``artifacts/`` directory."""
        path = self.artifacts_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_as_plain(payload), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return path

    def artifact_path(self, name: str) -> Path:
        """Absolute path for an artifact file, creating parent directories."""
        path = self.artifacts_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _as_plain(value: Any) -> Any:
    """Recursively convert dataclasses, numpy scalars and arrays to plain Python."""
    import dataclasses

    if is_dataclass(value) and not isinstance(value, type):
        return {k: _as_plain(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value
