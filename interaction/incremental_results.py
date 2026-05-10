"""
Incremental, resumable per-sample CSV writer.

Why this exists:
  Colab sessions die. A pilot of 1080 samples per system at ~10 s/sample
  is 3+ hours; if we write the CSV only at the end, a disconnect wipes
  the whole run. This module appends after every sample and provides a
  `done_sample_ids()` helper so the script can skip already-completed
  samples on resume.

Usage:
    writer = IncrementalCSVWriter(
        path=Path("results/tables/axis2_per_sample.csv"),
        columns=["sample_id", "affordance", "prompt", "kld", "sim", "nss"],
    )
    done = writer.done_sample_ids()  # call at start of run
    for sample in samples:
        if sample.id in done:
            continue
        ...do the work...
        writer.append({"sample_id": sample.id, ...})

The file is opened in append mode and flushed after every row, so it is
durable across session crashes. The header is written only on first
creation.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, Set


class IncrementalCSVWriter:
    """Append-only CSV writer with resume support."""

    def __init__(self, path: Path | str, columns: Iterable[str]):
        self.path = Path(path)
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None

    def _open(self):
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.columns)
        if is_new:
            self._writer.writeheader()
            self._file.flush()
            os.fsync(self._file.fileno())

    def append(self, row: Dict[str, object]):
        if self._writer is None:
            self._open()
        # Format floats with full precision but predictable shape
        clean = {}
        for k in self.columns:
            v = row.get(k, "")
            if isinstance(v, float):
                clean[k] = f"{v:.6f}"
            else:
                clean[k] = v
        self._writer.writerow(clean)
        self._file.flush()
        os.fsync(self._file.fileno())

    def done_sample_ids(self, key: str = "sample_id") -> Set[str]:
        """Return the set of sample_ids already present in the CSV."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return set()
        done = set()
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if key in row and row[key]:
                    done.add(row[key])
        return done

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
