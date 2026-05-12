"""SQLite setup and repository methods."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Generator

from data.models import (
    ConsentRecord,
    JournalEntry,
    Measurement,
    PatientProfile,
    PlanReviewStatus,
    RehabPlan,
    RedFlagEvent,
    SessionRecord,
    SwellingLevel,
)

def _resolve_db_path() -> str:
    """
    Resolve the SQLite file path from DATABASE_URL or DB_PATH.
    Supports bare paths ("acl_rehab.db") and SQLite URIs ("sqlite:///acl_rehab.db").
    Raises ValueError for unsupported URL schemes (e.g. postgresql://).
    """
    raw = os.getenv("DATABASE_URL") or os.getenv("DB_PATH", "./acl_rehab.db")
    if raw.startswith("sqlite:///"):
        return raw[len("sqlite:///"):]
    if "://" not in raw:          # bare file path or ":memory:"
        return raw
    raise ValueError(
        f"DATABASE_URL scheme not supported: {raw!r}. "
        "Only SQLite paths are accepted. See DEPLOYMENT.md for migration options."
    )


DB_PATH: str = _resolve_db_path()

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT    NOT NULL,
    side                 TEXT    NOT NULL,
    graft_type           TEXT    NOT NULL,
    surgery_date         TEXT    NOT NULL,
    weight_bearing_status TEXT   NOT NULL,
    meniscal_repair      TEXT    NOT NULL,
    stated_goal_text     TEXT    NOT NULL,
    protocol             TEXT    NOT NULL,
    pt_code              TEXT,
    created_at           TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id           INTEGER NOT NULL REFERENCES patients(id),
    date                 TEXT    NOT NULL,
    week_number          INTEGER NOT NULL,
    pain_score           INTEGER NOT NULL,
    swelling             TEXT    NOT NULL,
    giving_way           INTEGER NOT NULL,
    exercises_completed  TEXT    NOT NULL DEFAULT '[]',
    exercises_skipped    TEXT    NOT NULL DEFAULT '[]',
    session_notes        TEXT    NOT NULL DEFAULT '',
    duration_minutes     INTEGER
);

CREATE TABLE IF NOT EXISTS measurements (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   INTEGER NOT NULL REFERENCES patients(id),
    session_id   INTEGER REFERENCES sessions(id),
    measured_at  TEXT    NOT NULL,
    metric       TEXT    NOT NULL,
    value        REAL    NOT NULL,
    unit         TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS rehab_plans (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id              INTEGER NOT NULL REFERENCES patients(id),
    consent_record_id       INTEGER NOT NULL REFERENCES consent_records(id),
    generated_at            TEXT    NOT NULL,
    protocol                TEXT    NOT NULL,
    week_start              INTEGER NOT NULL,
    week_end                INTEGER NOT NULL,
    exercises               TEXT    NOT NULL DEFAULT '[]',
    model_used              TEXT    NOT NULL,
    rag_sources             TEXT    NOT NULL DEFAULT '[]',
    week_summary            TEXT    NOT NULL DEFAULT '',
    pt_flag_notes           TEXT    NOT NULL DEFAULT '',
    goal_protocol_conflicts TEXT    NOT NULL DEFAULT '[]',
    review_status           TEXT    NOT NULL DEFAULT 'pending',
    reviewed_at             TEXT,
    pt_review_notes         TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS consent_records (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id     INTEGER NOT NULL REFERENCES patients(id),
    consented_at   TEXT    NOT NULL,
    consent_type   TEXT    NOT NULL,
    model_used     TEXT    NOT NULL,
    data_sent_hash TEXT    NOT NULL,
    revoked_at     TEXT
);

CREATE TABLE IF NOT EXISTS red_flag_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id     INTEGER NOT NULL REFERENCES patients(id),
    triggered_at   TEXT    NOT NULL,
    flags          TEXT    NOT NULL DEFAULT '[]',
    pain_score     INTEGER NOT NULL,
    swelling       TEXT    NOT NULL,
    giving_way     INTEGER NOT NULL,
    reviewed_by_pt INTEGER NOT NULL DEFAULT 0,
    escalated      INTEGER NOT NULL DEFAULT 0,
    resolved_at    TEXT
);

CREATE TABLE IF NOT EXISTS journal_entries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    date       TEXT    NOT NULL,
    ciphertext BLOB    NOT NULL
);

CREATE TABLE IF NOT EXISTS patient_states (
    patient_id  INTEGER PRIMARY KEY REFERENCES patients(id),
    state       TEXT    NOT NULL DEFAULT 'onboarding',
    entered_at  TEXT    NOT NULL,
    trigger     TEXT    NOT NULL DEFAULT 'initial'
);

CREATE TABLE IF NOT EXISTS state_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  INTEGER NOT NULL REFERENCES patients(id),
    from_state  TEXT    NOT NULL,
    to_state    TEXT    NOT NULL,
    trigger     TEXT    NOT NULL,
    occurred_at TEXT    NOT NULL
);
"""


class Database:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row

    # ── PatientProfile ────────────────────────────────────────────────────────

    def save_patient(self, p: PatientProfile) -> int:
        cur = self._conn.execute(
            """INSERT INTO patients
               (name, side, graft_type, surgery_date, weight_bearing_status,
                meniscal_repair, stated_goal_text, protocol, pt_code, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                p.name,
                p.side,
                p.graft_type,
                str(p.surgery_date),
                p.weight_bearing_status,
                p.meniscal_repair,
                p.stated_goal_text,
                p.protocol,
                p.pt_code,
                p.created_at.isoformat(),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_patient(self, patient_id: int) -> PatientProfile:
        row = self._conn.execute(
            "SELECT * FROM patients WHERE id = ?", (patient_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Patient {patient_id} not found.")
        return PatientProfile(**dict(row))

    def get_all_patients(self) -> list[PatientProfile]:
        rows = self._conn.execute("SELECT * FROM patients ORDER BY name").fetchall()
        return [PatientProfile(**dict(r)) for r in rows]

    def update_protocol(self, patient_id: int, protocol: str) -> None:
        self._conn.execute(
            "UPDATE patients SET protocol = ? WHERE id = ?", (protocol, patient_id)
        )
        self._conn.commit()

    # ── SessionRecord ─────────────────────────────────────────────────────────

    def save_session(self, s: SessionRecord) -> int:
        cur = self._conn.execute(
            """INSERT INTO sessions
               (patient_id, date, week_number, pain_score, swelling, giving_way,
                exercises_completed, exercises_skipped, session_notes, duration_minutes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                s.patient_id,
                str(s.date),
                s.week_number,
                s.pain_score,
                s.swelling,
                int(s.giving_way),
                json.dumps(s.exercises_completed),
                json.dumps(s.exercises_skipped),
                s.session_notes,
                s.duration_minutes,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_sessions(self, patient_id: int) -> list[SessionRecord]:
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE patient_id = ? ORDER BY date", (patient_id,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["exercises_completed"] = json.loads(d["exercises_completed"])
            d["exercises_skipped"] = json.loads(d["exercises_skipped"])
            d["giving_way"] = bool(d["giving_way"])
            result.append(SessionRecord(**d))
        return result

    # ── Measurement ───────────────────────────────────────────────────────────

    def save_measurement(self, m: Measurement) -> int:
        cur = self._conn.execute(
            """INSERT INTO measurements
               (patient_id, session_id, measured_at, metric, value, unit)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (m.patient_id, m.session_id, m.measured_at.isoformat(), m.metric, m.value, m.unit),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_measurements(self, patient_id: int, metric: str | None = None) -> list[Measurement]:
        if metric:
            rows = self._conn.execute(
                "SELECT * FROM measurements WHERE patient_id = ? AND metric = ? ORDER BY measured_at",
                (patient_id, metric.lower()),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM measurements WHERE patient_id = ? ORDER BY measured_at",
                (patient_id,),
            ).fetchall()
        return [Measurement(**dict(r)) for r in rows]

    # ── ConsentRecord ─────────────────────────────────────────────────────────

    def save_consent(self, c: ConsentRecord) -> int:
        cur = self._conn.execute(
            """INSERT INTO consent_records
               (patient_id, consented_at, consent_type, model_used, data_sent_hash, revoked_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                c.patient_id,
                c.consented_at.isoformat(),
                c.consent_type,
                c.model_used,
                c.data_sent_hash,
                c.revoked_at.isoformat() if c.revoked_at else None,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_active_consent(self, patient_id: int, consent_type: str) -> ConsentRecord | None:
        row = self._conn.execute(
            """SELECT * FROM consent_records
               WHERE patient_id = ? AND consent_type = ? AND revoked_at IS NULL
               ORDER BY consented_at DESC LIMIT 1""",
            (patient_id, consent_type),
        ).fetchone()
        return ConsentRecord(**dict(row)) if row else None

    # ── RehabPlan ─────────────────────────────────────────────────────────────

    def save_rehab_plan(self, plan: RehabPlan) -> int:
        cur = self._conn.execute(
            """INSERT INTO rehab_plans
               (patient_id, consent_record_id, generated_at, protocol, week_start,
                week_end, exercises, model_used, rag_sources, week_summary,
                pt_flag_notes, goal_protocol_conflicts, review_status,
                reviewed_at, pt_review_notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                plan.patient_id,
                plan.consent_record_id,
                plan.generated_at.isoformat(),
                plan.protocol,
                plan.week_start,
                plan.week_end,
                json.dumps(plan.exercises),
                plan.model_used,
                json.dumps(plan.rag_sources),
                plan.week_summary,
                plan.pt_flag_notes,
                json.dumps(plan.goal_protocol_conflicts),
                plan.review_status,
                plan.reviewed_at.isoformat() if plan.reviewed_at else None,
                plan.pt_review_notes,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_latest_plan(self, patient_id: int) -> RehabPlan | None:
        row = self._conn.execute(
            "SELECT * FROM rehab_plans WHERE patient_id = ? ORDER BY generated_at DESC LIMIT 1",
            (patient_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_plan(dict(row))

    def get_plan(self, plan_id: int) -> RehabPlan | None:
        row = self._conn.execute(
            "SELECT * FROM rehab_plans WHERE id = ?", (plan_id,)
        ).fetchone()
        return _row_to_plan(dict(row)) if row else None

    def get_pending_plans(self) -> list[tuple[RehabPlan, PatientProfile]]:
        rows = self._conn.execute(
            """SELECT rp.*, p.name, p.protocol as pt_protocol, p.side, p.graft_type,
                      p.surgery_date, p.weight_bearing_status, p.meniscal_repair,
                      p.stated_goal_text, p.pt_code, p.created_at as pt_created_at
               FROM rehab_plans rp
               JOIN patients p ON rp.patient_id = p.id
               WHERE rp.review_status = 'pending'
               ORDER BY rp.generated_at DESC"""
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            # split into plan fields vs patient fields
            patient = PatientProfile(
                id=d["patient_id"],
                name=d["name"],
                side=d["side"],
                graft_type=d["graft_type"],
                surgery_date=d["surgery_date"],
                weight_bearing_status=d["weight_bearing_status"],
                meniscal_repair=d["meniscal_repair"],
                stated_goal_text=d["stated_goal_text"],
                protocol=d["pt_protocol"],
                pt_code=d.get("pt_code"),
                created_at=d["pt_created_at"],
            )
            plan = _row_to_plan(d)
            result.append((plan, patient))
        return result

    def approve_plan(self, plan_id: int, pt_notes: str = "") -> None:
        self._conn.execute(
            """UPDATE rehab_plans
               SET review_status = 'approved', reviewed_at = ?, pt_review_notes = ?
               WHERE id = ?""",
            (datetime.utcnow().isoformat(), pt_notes, plan_id),
        )
        self._conn.commit()

    def reject_plan(self, plan_id: int, pt_notes: str) -> None:
        self._conn.execute(
            """UPDATE rehab_plans
               SET review_status = 'rejected', reviewed_at = ?, pt_review_notes = ?
               WHERE id = ?""",
            (datetime.utcnow().isoformat(), pt_notes, plan_id),
        )
        self._conn.commit()

    def update_plan_exercises(self, plan_id: int, exercises: list, week_summary: str) -> None:
        """PT edit — replace exercises and summary before approving."""
        self._conn.execute(
            "UPDATE rehab_plans SET exercises = ?, week_summary = ? WHERE id = ?",
            (json.dumps(exercises), week_summary, plan_id),
        )
        self._conn.commit()

    # ── RedFlagEvent ──────────────────────────────────────────────────────────

    def save_red_flag(self, flag: RedFlagEvent) -> int:
        cur = self._conn.execute(
            """INSERT INTO red_flag_events
               (patient_id, triggered_at, flags, pain_score, swelling,
                giving_way, reviewed_by_pt, escalated, resolved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                flag.patient_id,
                flag.triggered_at.isoformat(),
                json.dumps(flag.flags),
                flag.pain_score,
                flag.swelling,
                int(flag.giving_way),
                int(flag.reviewed_by_pt),
                int(flag.escalated),
                flag.resolved_at.isoformat() if flag.resolved_at else None,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_open_red_flags(self, patient_id: int | None = None) -> list[RedFlagEvent]:
        if patient_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM red_flag_events WHERE patient_id = ? AND reviewed_by_pt = 0 ORDER BY triggered_at DESC",
                (patient_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM red_flag_events WHERE reviewed_by_pt = 0 ORDER BY triggered_at DESC"
            ).fetchall()
        return [_row_to_red_flag(dict(r)) for r in rows]

    def mark_flag_reviewed(self, flag_id: int) -> None:
        self._conn.execute(
            "UPDATE red_flag_events SET reviewed_by_pt = 1 WHERE id = ?", (flag_id,)
        )
        self._conn.commit()

    def escalate_flag(self, flag_id: int) -> None:
        self._conn.execute(
            "UPDATE red_flag_events SET escalated = 1, reviewed_by_pt = 1 WHERE id = ?", (flag_id,)
        )
        self._conn.commit()

    # ── JournalEntry ──────────────────────────────────────────────────────────
    # No plaintext is ever written here — ciphertext only. See data/journal.py.

    def save_journal_entry(self, entry: JournalEntry) -> int:
        cur = self._conn.execute(
            "INSERT INTO journal_entries (patient_id, date, ciphertext) VALUES (?, ?, ?)",
            (entry.patient_id, str(entry.date), entry.ciphertext),
        )
        self._conn.commit()
        return cur.lastrowid

    # ── Patient state (FSM) ───────────────────────────────────────────────────

    def get_patient_state(self, patient_id: int) -> str:
        row = self._conn.execute(
            "SELECT state FROM patient_states WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        return row["state"] if row else "onboarding"

    def set_patient_state(self, patient_id: int, state: str, trigger: str) -> None:
        old_state = self.get_patient_state(patient_id)
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """INSERT INTO patient_states (patient_id, state, entered_at, trigger)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(patient_id) DO UPDATE SET
                 state = excluded.state,
                 entered_at = excluded.entered_at,
                 trigger = excluded.trigger""",
            (patient_id, state, now, trigger),
        )
        self._conn.execute(
            """INSERT INTO state_history
               (patient_id, from_state, to_state, trigger, occurred_at)
               VALUES (?, ?, ?, ?, ?)""",
            (patient_id, old_state, state, trigger, now),
        )
        self._conn.commit()

    def get_state_history(self, patient_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM state_history WHERE patient_id = ? ORDER BY occurred_at",
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── JournalEntry ──────────────────────────────────────────────────────────

    def get_journal_entries_raw(self, patient_id: int) -> list[JournalEntry]:
        rows = self._conn.execute(
            "SELECT * FROM journal_entries WHERE patient_id = ? ORDER BY date",
            (patient_id,),
        ).fetchall()
        return [JournalEntry(**dict(r)) for r in rows]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row_to_plan(d: dict) -> RehabPlan:
    d["exercises"] = json.loads(d["exercises"])
    d["rag_sources"] = json.loads(d["rag_sources"])
    d["goal_protocol_conflicts"] = json.loads(d.get("goal_protocol_conflicts", "[]"))
    return RehabPlan(**{k: v for k, v in d.items() if k in RehabPlan.model_fields})


def _row_to_red_flag(d: dict) -> RedFlagEvent:
    d["flags"] = json.loads(d["flags"])
    d["giving_way"] = bool(d["giving_way"])
    d["reviewed_by_pt"] = bool(d["reviewed_by_pt"])
    d["escalated"] = bool(d["escalated"])
    return RedFlagEvent(**d)


@contextmanager
def get_db() -> Generator[Database, None, None]:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    try:
        yield Database(conn)
    finally:
        conn.close()
