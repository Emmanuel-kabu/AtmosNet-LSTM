"""Data-lake helpers — Parquet-partitioned storage conventions.

Folder layout
─────────────
    data/lake/raw/date=YYYY-MM-DD/part-0.parquet
    data/lake/clean/date=YYYY-MM-DD/part-0.parquet
    data/lake/features/date=YYYY-MM-DD/part-0.parquet

Rules:
* **raw** partitions are immutable (append-only, never overwrite).
* **clean** and **features** can be regenerated from raw.
* ``pyarrow`` is the only I/O backend — no Spark dependency.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ── Public constants ──────────────────────────────────────────────────────
LAYER_RAW = "raw"
LAYER_CLEAN = "clean"
LAYER_FEATURES = "features"
LAYERS = (LAYER_RAW, LAYER_CLEAN, LAYER_FEATURES)


# ── helpers ───────────────────────────────────────────────────────────────

def partition_dir(lake_root: Path, layer: str, partition_date: date) -> Path:
    """Return ``lake_root/<layer>/date=YYYY-MM-DD``."""
    return lake_root / layer / f"date={partition_date.isoformat()}"


def ensure_lake_dirs(lake_root: Path) -> None:
    """Create the three layer directories if they don't exist."""
    for layer in LAYERS:
        (lake_root / layer).mkdir(parents=True, exist_ok=True)
    logger.debug("Lake directories verified at %s", lake_root)


# ── Write ─────────────────────────────────────────────────────────────────

def write_partition(
    df: pd.DataFrame,
    lake_root: Path,
    layer: str,
    partition_date: date,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a DataFrame as a single Parquet file inside a date partition.

    Parameters
    ----------
    df : pd.DataFrame
        Data to persist.
    lake_root : Path
        Root of the data lake (e.g. ``data/lake``).
    layer : str
        One of ``"raw"``, ``"clean"``, ``"features"``.
    partition_date : date
        Partition key.
    overwrite : bool
        If *True*, replace existing partition (forbidden for ``raw``).

    Returns
    -------
    Path
        Path to the written Parquet file.
    """
    if layer not in LAYERS:
        raise ValueError(f"Unknown layer {layer!r}. Must be one of {LAYERS}")

    if layer == LAYER_RAW and overwrite:
        raise ValueError("Raw partitions are immutable — set overwrite=False.")

    part_dir = partition_dir(lake_root, layer, partition_date)
    part_dir.mkdir(parents=True, exist_ok=True)
    out_path = part_dir / "part-0.parquet"

    if out_path.exists() and not overwrite:
        if layer == LAYER_RAW:
            logger.info("Raw partition %s already exists — skipping", partition_date)
            return out_path
        # For clean/features we may want to regenerate
        logger.warning("Partition %s/%s exists — overwriting", layer, partition_date)

    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, out_path, compression="snappy")
    logger.info("Wrote %d rows → %s", len(df), out_path)
    return out_path


# ── Read ──────────────────────────────────────────────────────────────────

def read_partition(
    lake_root: Path,
    layer: str,
    partition_date: date,
) -> pd.DataFrame:
    """Read a single date partition back into a DataFrame."""
    part_dir = partition_dir(lake_root, layer, partition_date)
    parquet_file = part_dir / "part-0.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Partition not found: {parquet_file}")
    return pq.read_table(parquet_file).to_pandas()


def read_date_range(
    lake_root: Path,
    layer: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Read all partitions in [start, end] and concatenate."""
    frames: list[pd.DataFrame] = []
    current = start
    from datetime import timedelta

    while current <= end:
        part_dir = partition_dir(lake_root, layer, current)
        parquet_file = part_dir / "part-0.parquet"
        if parquet_file.exists():
            frames.append(pq.read_table(parquet_file).to_pandas())
        current += timedelta(days=1)

    if not frames:
        raise FileNotFoundError(
            f"No partitions found in {layer} between {start} and {end}"
        )
    return pd.concat(frames, ignore_index=False).sort_index()


def read_all_partitions(
    lake_root: Path,
    layer: str,
) -> pd.DataFrame:
    """Read every partition in a layer (full-history load)."""
    layer_dir = lake_root / layer
    if not layer_dir.exists():
        raise FileNotFoundError(f"Layer directory not found: {layer_dir}")

    parquet_files = sorted(layer_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {layer_dir}")

    table = pq.read_table(layer_dir, use_pandas_metadata=True)
    return table.to_pandas()


def list_partition_dates(lake_root: Path, layer: str) -> list[date]:
    """Return sorted list of date partitions that exist for a layer."""
    layer_dir = lake_root / layer
    if not layer_dir.exists():
        return []

    dates: list[date] = []
    for child in sorted(layer_dir.iterdir()):
        if child.is_dir() and child.name.startswith("date="):
            try:
                dates.append(date.fromisoformat(child.name.removeprefix("date=")))
            except ValueError:
                logger.warning("Skipping malformed partition dir: %s", child.name)
    return dates


def pending_partitions(
    lake_root: Path,
    source_layer: str,
    target_layer: str,
) -> list[date]:
    """Return dates present in *source_layer* but missing from *target_layer*."""
    source = set(list_partition_dates(lake_root, source_layer))
    target = set(list_partition_dates(lake_root, target_layer))
    return sorted(source - target)
