"""DPO training-pair export manager (Phase 2.3).

Turns pending corrections into TRL DPOTrainer-compatible JSONL files, one
file per tenant, with monotonic version numbering.  Graduation is atomic:
the JSONL is written first; only on success are corrections marked
'graduated' and training_pairs rows inserted in a single transaction.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from accord_ai.feedback.dpo_prompt import (
    DPO_TEMPLATE_VERSION,
    render_prompt,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema-fragment walker
# ---------------------------------------------------------------------------


def _schema_fragment(field_path: str) -> dict:
    """Return the JSON-schema sub-object for field_path from CustomerSubmission.

    Walks `properties` keys, resolving top-level $defs references.
    Falls back to {"type": "string"} on any failure (schema drift is expected
    as the domain model evolves).
    """
    try:
        from accord_ai.schema import CustomerSubmission
        schema = CustomerSubmission.model_json_schema()
        defs = schema.get("$defs", {})
        node: dict = schema
        for part in field_path.split("."):
            props = node.get("properties", {})
            if part not in props:
                return {"type": "string"}
            node = props[part]
            # Resolve a $ref one level — nested refs are unusual in our schema.
            if "$ref" in node:
                ref_key = node["$ref"].rsplit("/", 1)[-1]
                node = defs.get(ref_key, {"type": "string"})
        return node
    except Exception:
        return {"type": "string"}


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


@contextmanager
def _get_conn(db_path: str) -> Iterator[sqlite3.Connection]:
    """Open a fresh connection, yield, commit on success, rollback on error."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DPOPair:
    prompt: str
    chosen: str
    rejected: str
    field_path: str
    correction_id: str
    template_version: str
    explanation: Optional[str] = None


@dataclass(frozen=True)
class ExportResult:
    tenant: str
    path: Optional[Path]
    count: int


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class DPOManager:
    """Build and export DPO training pairs from pending corrections.

    Thread-safe: each public method opens and closes its own connection via
    _get_conn (no shared state beyond the db_path and output_dir paths).
    """

    def __init__(
        self,
        db_path: str,
        output_dir: Path,
        threshold: int = 50,
    ) -> None:
        self._db_path = db_path
        self._output_dir = Path(output_dir)
        self._threshold = threshold

    # --- Eligibility --------------------------------------------------------

    def count_pending(self, tenant: str) -> int:
        with _get_conn(self._db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM corrections WHERE tenant=? AND status='pending'",
                (tenant,),
            ).fetchone()
        return row[0] if row else 0

    def eligible_for_training(self, tenant: str) -> bool:
        return self.count_pending(tenant) >= self._threshold

    # --- Pair building -------------------------------------------------------

    def build_pairs(self, tenant: str, limit: Optional[int] = None) -> List[DPOPair]:
        """Fetch pending corrections, build DPO pairs. No DB mutation.

        Message lookup is done per-correction (N+1 queries) because SQLite
        does not allow correlated outer-column references inside LIMIT/OFFSET
        of a subquery.  Export is a batch/CLI operation — latency is not a
        concern.
        """
        correction_sql = """
            SELECT id, session_id, turn, field_path,
                   wrong_value_json, correct_value_json, explanation
            FROM corrections
            WHERE tenant = ? AND status = 'pending'
            ORDER BY created_at ASC
        """
        if limit is not None:
            correction_sql += f" LIMIT {int(limit)}"

        msg_sql = """
            SELECT content FROM messages
            WHERE session_id = ? AND role = 'user'
            ORDER BY created_at ASC, message_id ASC
            LIMIT 1 OFFSET ?
        """

        with _get_conn(self._db_path) as conn:
            c_rows = conn.execute(correction_sql, (tenant,)).fetchall()
            enriched = []
            for row in c_rows:
                turn = row["turn"]
                offset = max(0, turn - 1) if turn > 0 else 0
                msg = conn.execute(msg_sql, (row["session_id"], offset)).fetchone()
                enriched.append((row, msg[0] if msg else None))

        pairs: List[DPOPair] = []
        for row, user_text in enriched:
            pair = self._build_pair(row, user_text)
            if pair is not None:
                pairs.append(pair)
        return pairs

    def _build_pair(self, row: sqlite3.Row, user_text: Optional[str]) -> Optional[DPOPair]:
        correction_id = row["id"]
        field_path = row["field_path"]

        if user_text is None:
            _logger.warning(
                "dpo_skipped_missing_message correction_id=%s session_id=%s turn=%s",
                correction_id, row["session_id"], row["turn"],
            )
            return None

        wrong_raw = row["wrong_value_json"]
        correct_raw = row["correct_value_json"]
        if wrong_raw == correct_raw:
            _logger.warning(
                "dpo_skipped_identical_values correction_id=%s field=%s",
                correction_id, field_path,
            )
            return None

        leaf = field_path.split(".")[-1]
        wrong_val = json.loads(wrong_raw) if wrong_raw else None
        correct_val = json.loads(correct_raw) if correct_raw else None

        fragment = _schema_fragment(field_path)
        prompt = render_prompt(
            field_path=field_path,
            schema_json=json.dumps(fragment),
            user_text=user_text,
        )
        chosen = json.dumps({leaf: correct_val})
        rejected = json.dumps({leaf: wrong_val})

        return DPOPair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            field_path=field_path,
            correction_id=correction_id,
            template_version=DPO_TEMPLATE_VERSION,
            explanation=row["explanation"],
        )

    # --- Export --------------------------------------------------------------

    def export(self, tenant: str) -> ExportResult:
        """Build pairs, write JSONL, mark graduated atomically."""
        pairs = self.build_pairs(tenant)
        if not pairs:
            return ExportResult(tenant=tenant, path=None, count=0)

        tenant_dir = self._output_dir / tenant
        tenant_dir.mkdir(parents=True, exist_ok=True)
        version = self._next_version(tenant_dir)
        path = tenant_dir / f"dpo_v{version}.jsonl"

        # Write JSONL first — if this fails, no DB mutation happens.
        with path.open("w") as fh:
            for p in pairs:
                fh.write(
                    json.dumps({
                        "prompt": p.prompt,
                        "chosen": p.chosen,
                        "rejected": p.rejected,
                        "metadata": {
                            "field_path": p.field_path,
                            "correction_id": p.correction_id,
                            "template_version": p.template_version,
                        },
                    }) + "\n"
                )

        # DB mutation only after successful write.
        self._graduate(tenant, pairs, path)

        return ExportResult(tenant=tenant, path=path, count=len(pairs))

    def _graduate(self, tenant: str, pairs: List[DPOPair], export_path: Path) -> None:
        """Atomic: mark corrections graduated + insert training_pairs rows."""
        now = datetime.now(timezone.utc).isoformat()
        export_path_str = str(export_path)
        with _get_conn(self._db_path) as conn:
            for pair in pairs:
                conn.execute(
                    "UPDATE corrections SET status='graduated', graduated_at=? "
                    "WHERE id=? AND tenant=?",
                    (now, pair.correction_id, tenant),
                )
                conn.execute(
                    """
                    INSERT INTO training_pairs
                        (id, tenant, prompt, chosen, rejected, field_path,
                         correction_id, explanation, status, created_at, exported_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'exported', ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        tenant,
                        "<prompt truncated in DB>",
                        "<chosen in JSONL>",
                        "<rejected in JSONL>",
                        pair.field_path,
                        pair.correction_id,
                        pair.explanation,
                        now,
                        export_path_str,
                    ),
                )

    def _next_version(self, tenant_dir: Path) -> int:
        existing = list(tenant_dir.glob("dpo_v*.jsonl"))
        if not existing:
            return 1
        versions = [int(p.stem.removeprefix("dpo_v")) for p in existing]
        return max(versions) + 1

    # --- Status query --------------------------------------------------------

    def status(self, tenant: str) -> dict:
        """Return counts and last export info for a tenant."""
        with _get_conn(self._db_path) as conn:
            pending = conn.execute(
                "SELECT COUNT(*) FROM corrections WHERE tenant=? AND status='pending'",
                (tenant,),
            ).fetchone()[0]
            graduated = conn.execute(
                "SELECT COUNT(*) FROM corrections WHERE tenant=? AND status='graduated'",
                (tenant,),
            ).fetchone()[0]
            last = conn.execute(
                "SELECT exported_at, prompt FROM training_pairs "
                "WHERE tenant=? ORDER BY created_at DESC LIMIT 1",
                (tenant,),
            ).fetchone()
        last_export_path = last["prompt"] if last else None
        last_export_at = last["exported_at"] if last else None
        return {
            "tenant": tenant,
            "pending": pending,
            "graduated": graduated,
            "eligible_for_training": pending >= self._threshold,
            "last_export_path": last_export_path,
            "last_export_at": last_export_at,
        }

    def list_tenants_with_pending(self) -> List[str]:
        """All tenants that have at least one pending correction."""
        with _get_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT tenant FROM corrections WHERE status='pending'"
            ).fetchall()
        return [r["tenant"] for r in rows]
