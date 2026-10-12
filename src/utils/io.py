"""Experiment persistence: run records, registry, logging.

Design rules:
- every run gets a unique id and its own JSON file; files are never
  overwritten (write_run_record refuses to clobber);
- a flat registry CSV (results/index.csv) is appended to, one row per run;
- failures are recorded with status='failed' and a reason, never silently
  discarded.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

RESULTS_DIR = Path("results")
RUNS_DIR = RESULTS_DIR / "runs"
SERIES_DIR = RESULTS_DIR / "series"


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


def experiment_metadata(code_version: str = "", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    md = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version,
    }
    if extra:
        md.update(extra)
    return md


def write_run_record(record: Dict[str, Any], results_dir: Path = RESULTS_DIR) -> Path:
    """Write a run record as JSON. Refuses to overwrite existing files."""
    runs = Path(results_dir) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    rid = record.get("run_id")
    if not rid:
        raise ValueError("run record must contain 'run_id'")
    path = runs / f"{rid}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing run record: {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, sort_keys=True, default=float)
    _append_index_row(record, results_dir)
    return path


def _append_index_row(record: Dict[str, Any], results_dir: Path) -> None:
    idx = Path(results_dir) / "index.csv"
    row = {
        "run_id": record.get("run_id"),
        "status": record.get("status"),
        "experiment": record.get("experiment"),
        "scenario": record.get("scenario", {}).get("name") if isinstance(record.get("scenario"), dict) else record.get("scenario"),
        "controller": record.get("controller"),
        "seed": record.get("seed"),
        "duration_s": record.get("duration_s"),
    }
    m = record.get("metrics") or {}
    for key in ("goodput_mbps", "mean_rtt_ms", "p95_rtt_ms", "loss_rate", "jain_fairness"):
        row[key] = m.get(key)
    idx.parent.mkdir(parents=True, exist_ok=True)
    header = not idx.exists()
    with open(idx, "a", encoding="utf-8") as f:
        if header:
            f.write(",".join(row.keys()) + "\n")
        else:
            # keep header stable: only write keys present in header
            with open(idx, "r", encoding="utf-8") as hf:
                cols = hf.readline().strip().split(",")
            ordered = {k: row.get(k) for k in cols}
            row = ordered
        vals = []
        for v in row.values():
            if v is None:
                vals.append("")
            else:
                s = str(v)
                if any(ch in s for ch in ",\n\""):
                    s = '"' + s.replace('"', '""') + '"'
                vals.append(s)
        f.write(",".join(vals) + "\n")


def write_time_series(run_id: str, series: Dict[str, Any], results_dir: Path = RESULTS_DIR) -> Path:
    """Persist detailed per-bucket time series for a run (JSON, gzipped later if needed)."""
    d = Path(results_dir) / "series"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{run_id}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite series file: {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(series, f, sort_keys=True, default=float)
    return path


def load_run_records(results_dir: Path = RESULTS_DIR, experiment: Optional[str] = None,
                     status: Optional[str] = "ok") -> list:
    """Load all run-record JSONs, optionally filtered."""
    import pandas as pd  # local import: only needed for analysis
    runs = Path(results_dir) / "runs"
    rows = []
    for p in sorted(runs.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if status is not None and rec.get("status") != status:
            continue
        if experiment is not None and rec.get("experiment") != experiment:
            continue
        rows.append(rec)
    return rows


def records_to_dataframe(records: list) -> "pd.DataFrame":
    import pandas as pd
    flat = []
    for rec in records:
        row = {
            "run_id": rec.get("run_id"),
            "experiment": rec.get("experiment"),
            "scenario": (rec.get("scenario") or {}).get("name"),
            "controller": rec.get("controller"),
            "controller_variant": rec.get("controller_variant"),
            "seed": rec.get("seed"),
            "duration_s": rec.get("duration_s"),
            "warmup_s": (rec.get("scenario") or {}).get("warmup_s"),
        }
        sc = rec.get("scenario") or {}
        for k in ("bandwidth_mbps", "base_rtt_ms", "n_flows", "loss_rate", "queue_bdp_factor", "queue_packets"):
            row[k] = sc.get(k)
        row.update({f"m_{k}": v for k, v in (rec.get("metrics") or {}).items()})
        row["train_time_s"] = rec.get("training_time_s")
        row["model_info"] = json.dumps(rec.get("model_info") or {})
        flat.append(row)
    return pd.DataFrame(flat)


def git_commit() -> str:
    """Best-effort git commit hash (empty string if git unavailable)."""
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
