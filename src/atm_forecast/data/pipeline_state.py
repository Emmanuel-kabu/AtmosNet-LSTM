"""Pipeline-state watermark table backed by Postgres (or any SQLAlchemy DB).

Schema
──────
    pipeline_state
    ├── pipeline_name        VARCHAR(128)  PK
    ├── last_success_at      TIMESTAMP     — max observed_at from last run
    ├── last_partition_date   DATE          — most recent partition written
    ├── rows_ingested        BIGINT        — total rows in last batch
    └── updated_at           TIMESTAMP     — last mutation timestamp

Usage
─────
    from atm_forecast.data.pipeline_state import get_watermark, update_watermark

    wm = get_watermark(engine, "weather_ingest")
    # → wm.last_success_at is None on first run

    # ... ingest data ...

    update_watermark(engine, "weather_ingest",
                     last_success_at=max_ts, last_partition_date=today,
                     rows_ingested=len(df))
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    MetaData,
    String,
    Table,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

metadata_obj = MetaData()

pipeline_state = Table(
    "pipeline_state",
    metadata_obj,
    Column("pipeline_name", String(128), primary_key=True),
    Column("last_success_at", DateTime(timezone=True), nullable=True),
    Column("last_partition_date", Date, nullable=True),
    Column("rows_ingested", BigInteger, default=0),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


# ── Helpers ───────────────────────────────────────────────────────────────

def ensure_table(engine: Engine) -> None:
    """Create the ``pipeline_state`` table if it doesn't exist yet."""
    metadata_obj.create_all(engine, tables=[pipeline_state])
    logger.info("pipeline_state table ensured")


def get_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine."""
    return create_engine(database_url, future=True)


# ── dataclass-like result ─────────────────────────────────────────────────

class Watermark:
    """Read-only snapshot of a pipeline's watermark state."""

    __slots__ = ("pipeline_name", "last_success_at", "last_partition_date",
                 "rows_ingested", "updated_at")

    def __init__(
        self,
        pipeline_name: str,
        last_success_at: datetime | None = None,
        last_partition_date: date | None = None,
        rows_ingested: int = 0,
        updated_at: datetime | None = None,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.last_success_at = last_success_at
        self.last_partition_date = last_partition_date
        self.rows_ingested = rows_ingested
        self.updated_at = updated_at

    def __repr__(self) -> str:
        return (
            f"Watermark(pipeline={self.pipeline_name!r}, "
            f"last_success_at={self.last_success_at}, "
            f"last_partition={self.last_partition_date}, "
            f"rows={self.rows_ingested})"
        )


# ── CRUD ──────────────────────────────────────────────────────────────────

def get_watermark(engine: Engine, pipeline_name: str) -> Watermark:
    """Read the current watermark for *pipeline_name*.

    Returns a ``Watermark`` with ``last_success_at=None`` if no row exists
    (first run).
    """
    ensure_table(engine)
    stmt = select(pipeline_state).where(pipeline_state.c.pipeline_name == pipeline_name)

    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()

    if row is None:
        logger.info("No watermark found for %r — assuming first run", pipeline_name)
        return Watermark(pipeline_name=pipeline_name)

    return Watermark(
        pipeline_name=row["pipeline_name"],
        last_success_at=row["last_success_at"],
        last_partition_date=row["last_partition_date"],
        rows_ingested=row["rows_ingested"],
        updated_at=row["updated_at"],
    )


def update_watermark(
    engine: Engine,
    pipeline_name: str,
    *,
    last_success_at: datetime,
    last_partition_date: date,
    rows_ingested: int = 0,
) -> None:
    """Upsert the watermark for *pipeline_name*.

    Uses Postgres ``ON CONFLICT … DO UPDATE`` when available; otherwise falls
    back to a SELECT-then-INSERT/UPDATE pattern so SQLite works during dev.
    """
    ensure_table(engine)
    now = datetime.now(timezone.utc)
    values = {
        "pipeline_name": pipeline_name,
        "last_success_at": last_success_at,
        "last_partition_date": last_partition_date,
        "rows_ingested": rows_ingested,
        "updated_at": now,
    }

    dialect = engine.dialect.name

    with engine.begin() as conn:
        if dialect == "postgresql":
            stmt = pg_insert(pipeline_state).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["pipeline_name"],
                set_={
                    "last_success_at": stmt.excluded.last_success_at,
                    "last_partition_date": stmt.excluded.last_partition_date,
                    "rows_ingested": stmt.excluded.rows_ingested,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            conn.execute(stmt)
        else:
            # Generic fallback (SQLite, etc.)
            existing = conn.execute(
                select(pipeline_state.c.pipeline_name).where(
                    pipeline_state.c.pipeline_name == pipeline_name
                )
            ).first()
            if existing:
                conn.execute(
                    pipeline_state.update()
                    .where(pipeline_state.c.pipeline_name == pipeline_name)
                    .values(**{k: v for k, v in values.items() if k != "pipeline_name"})
                )
            else:
                conn.execute(pipeline_state.insert().values(**values))

    logger.info(
        "Watermark updated — %s: last_success_at=%s, partition=%s, rows=%d",
        pipeline_name, last_success_at, last_partition_date, rows_ingested,
    )
