"""Manifest writer — records what partitions were produced by each pipeline run.

A manifest is a small JSON file stored in ``data/manifests/`` that records:
* which pipeline ran
* which date partitions were written
* row counts and timing

Example manifest
────────────────
    {
      "pipeline": "weather_ingest",
      "run_id": "20260219T143022Z",
      "started_at": "2026-02-19T14:30:22+00:00",
      "finished_at": "2026-02-19T14:30:45+00:00",
      "partitions": [
        {"date": "2026-02-18", "layer": "raw", "rows": 24, "file": "data/lake/raw/date=2026-02-18/part-0.parquet"}
      ],
      "status": "success"
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PartitionEntry:
    """One partition written during a pipeline run."""

    date: str  # ISO date string
    layer: str
    rows: int
    file: str


@dataclass
class Manifest:
    """Full manifest for a single pipeline run."""

    pipeline: str
    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    partitions: list[PartitionEntry] = field(default_factory=list)
    status: str = "pending"

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat()

    # ── Mutation methods ──────────────────────────────────────────────

    def add_partition(
        self,
        partition_date: date,
        layer: str,
        rows: int,
        file_path: str | Path,
    ) -> None:
        self.partitions.append(
            PartitionEntry(
                date=partition_date.isoformat(),
                layer=layer,
                rows=rows,
                file=str(file_path),
            )
        )

    def mark_success(self) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.status = "success"

    def mark_failure(self, error: str = "") -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.status = f"failed: {error}" if error else "failed"

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, manifests_dir: str | Path) -> Path:
        """Write the manifest as a JSON file."""
        manifests_dir = Path(manifests_dir)
        manifests_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.pipeline}_{self.run_id}.json"
        path = manifests_dir / filename
        path.write_text(
            json.dumps(asdict(self), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Manifest saved → %s", path)
        return path


def load_manifest(path: str | Path) -> Manifest:
    """Load a manifest from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    partitions = [PartitionEntry(**p) for p in data.pop("partitions", [])]
    return Manifest(**data, partitions=partitions)
