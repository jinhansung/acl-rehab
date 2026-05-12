"""
Deterministic red-flag rule engine — v0.1 rules RF001–RF005.
No LLM is involved at any point. All decisions are pure threshold / boolean logic.

Rules
─────
RF001  Fever >38.0 °C AND (calf swelling OR calf pain)            [DVT / septic arthritis]
RF002  Sudden ROM loss >10 ° within 24 h                           [haemarthrosis / graft failure]
RF003  Mechanical locking (cannot complete normal ROM)             [loose body / meniscal block]
RF004  Wound change AND fever >38.0 °C                            [surgical-site infection]
RF005  Pain >7/10 sustained for ≥48 h                             [unresolved severe pain]

Each triggered rule returns a RedFlag with:
  rule_id          — "RF001" … "RF005"
  severity         — "URGENT" (all v0.1 rules) or "MONITOR"
  message          — patient-safe text (passes TONE_RULES)
  evidence_snapshot — dict of the exact field values that fired the rule

Backward compatibility
──────────────────────
All parameters that existed in the prior API are preserved with their original
defaults so existing callers (pages/2_daily_session.py, tests/) work unchanged.
  fever=True          → treated as temperature_celsius > 38.0 when no numeric temp given
  wound_drainage=True → mapped to wound_change=True (RF004 predecessor)
  giving_way          → accepted but fires no v0.1 rule; retained for callers
  swelling            → accepted but fires no v0.1 rule; retained for callers
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RedFlag:
    rule_id: str
    severity: str                        # "URGENT" | "MONITOR"
    message: str
    evidence_snapshot: dict[str, Any]

    def __str__(self) -> str:
        return self.message


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_fever(
    temperature_celsius: float | None,
    fever_bool: bool,
) -> tuple[bool, float | None]:
    """
    Return (is_fever, resolved_temp).
    Numeric temperature takes precedence over the boolean flag.
    """
    if temperature_celsius is not None:
        return temperature_celsius > 38.0, temperature_celsius
    if fever_bool:
        return True, None   # exact temperature not recorded
    return False, None


# ── Public API ────────────────────────────────────────────────────────────────

def check_red_flags(
    pain: int,
    swelling: str = "None",
    giving_way: bool = False,          # retained for callers; no v0.1 rule uses it
    # ── backward-compat params ─────────────────────────────────────────────
    fever: bool = False,               # use temperature_celsius when available
    wound_drainage: bool = False,      # maps to wound_change (RF004 predecessor)
    # ── v0.1 rule-specific params ──────────────────────────────────────────
    temperature_celsius: float | None = None,
    calf_swelling: bool = False,
    calf_pain: bool = False,
    rom_loss_degrees: float | None = None,
    mechanical_locking: bool = False,
    wound_change: bool = False,
    pain_duration_hours: float = 0.0,
) -> list[RedFlag]:
    """
    Evaluate all v0.1 red-flag rules. Returns one RedFlag per triggered rule.
    Empty list means no flags — safe to continue session.

    Order of returned flags matches rule numbering (RF001 first).
    Duplicates are impossible; each rule fires at most once per call.
    """
    # Normalise backward-compat aliases
    effective_wound_change = wound_change or wound_drainage
    has_fever, temp_val = _resolve_fever(temperature_celsius, fever)

    triggered: list[RedFlag] = []

    # ── RF001: Fever >38.0 °C with calf swelling or calf pain ────────────────
    if has_fever and (calf_swelling or calf_pain):
        triggered.append(RedFlag(
            rule_id="RF001",
            severity="URGENT",
            message=(
                "Fever combined with calf symptoms — possible DVT or septic arthritis. "
                "Contact your PT or seek same-day medical care."
            ),
            evidence_snapshot={
                "temperature_celsius": temp_val,
                "fever_flag": fever,
                "calf_swelling": calf_swelling,
                "calf_pain": calf_pain,
            },
        ))

    # ── RF002: Sudden ROM loss >10 ° within 24 hours ─────────────────────────
    if rom_loss_degrees is not None and rom_loss_degrees > 10.0:
        triggered.append(RedFlag(
            rule_id="RF002",
            severity="URGENT",
            message=(
                f"A sudden loss of {rom_loss_degrees:.0f}° of movement in 24 hours "
                "may indicate haemarthrosis or a graft concern. "
                "Stop exercises and contact your PT today."
            ),
            evidence_snapshot={
                "rom_loss_degrees": rom_loss_degrees,
                "pain": pain,
            },
        ))

    # ── RF003: Mechanical locking ─────────────────────────────────────────────
    if mechanical_locking:
        triggered.append(RedFlag(
            rule_id="RF003",
            severity="URGENT",
            message=(
                "Mechanical locking — the knee cannot move through its normal range. "
                "Do not force the joint. Contact your PT or attend urgent care."
            ),
            evidence_snapshot={
                "mechanical_locking": True,
                "pain": pain,
                "swelling": swelling,
            },
        ))

    # ── RF004: Wound change with fever >38.0 °C ───────────────────────────────
    if effective_wound_change and has_fever:
        triggered.append(RedFlag(
            rule_id="RF004",
            severity="URGENT",
            message=(
                "A wound change combined with fever raises concern for surgical-site infection. "
                "Seek medical evaluation today — do not wait for a routine appointment."
            ),
            evidence_snapshot={
                "wound_change": effective_wound_change,
                "temperature_celsius": temp_val,
                "fever_flag": fever,
            },
        ))

    # ── RF005: Pain >7/10 sustained for ≥48 hours ────────────────────────────
    if pain > 7 and pain_duration_hours >= 48.0:
        triggered.append(RedFlag(
            rule_id="RF005",
            severity="URGENT",
            message=(
                f"Pain at {pain}/10 has continued for "
                f"{pain_duration_hours:.0f} hours. "
                "A PT review is warranted before your next session."
            ),
            evidence_snapshot={
                "pain": pain,
                "pain_duration_hours": pain_duration_hours,
            },
        ))

    return triggered
