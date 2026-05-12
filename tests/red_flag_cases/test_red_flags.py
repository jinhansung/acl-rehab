"""
50 synthetic red-flag test cases for agent/red_flags.py v0.1 rules RF001–RF005.

Structure
─────────
TRUE_POSITIVES  — 20 cases (at least one rule must fire)
DISTRACTORS     — 30 cases (no rule should fire; boundary conditions)

Sensitivity assertion: all 20 true positives must trigger ≥ 1 flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agent.red_flags import RedFlag, check_red_flags


# ── Case container ─────────────────────────────────────────────────────────────

@dataclass
class RedFlagCase:
    description: str
    kwargs: dict[str, Any]
    expected_rule_ids: list[str] = field(default_factory=list)  # empty for distractors


# ── True positives (20) ────────────────────────────────────────────────────────
# Rule coverage: RF001×4, RF002×4, RF003×3, RF004×4, RF005×4, multi-rule×1

TRUE_POSITIVES: list[RedFlagCase] = [
    # ── RF001: Fever >38.0 °C AND calf swelling ──────────────────────────────
    RedFlagCase(
        "RF001 — numeric temp 38.1 °C with calf swelling",
        {"pain": 3, "temperature_celsius": 38.1, "calf_swelling": True},
        ["RF001"],
    ),
    RedFlagCase(
        "RF001 — numeric temp 39.5 °C with calf pain only",
        {"pain": 2, "temperature_celsius": 39.5, "calf_pain": True},
        ["RF001"],
    ),
    RedFlagCase(
        "RF001 — boolean fever=True (no numeric) with calf swelling",
        {"pain": 4, "fever": True, "calf_swelling": True},
        ["RF001"],
    ),
    RedFlagCase(
        "RF001 — boolean fever=True with both calf swelling and calf pain",
        {"pain": 1, "fever": True, "calf_swelling": True, "calf_pain": True},
        ["RF001"],
    ),

    # ── RF002: ROM loss >10 ° in 24 h ────────────────────────────────────────
    RedFlagCase(
        "RF002 — 11 ° ROM loss",
        {"pain": 5, "rom_loss_degrees": 11.0},
        ["RF002"],
    ),
    RedFlagCase(
        "RF002 — 15 ° ROM loss",
        {"pain": 2, "rom_loss_degrees": 15.0},
        ["RF002"],
    ),
    RedFlagCase(
        "RF002 — 10.1 ° ROM loss (just above threshold)",
        {"pain": 3, "rom_loss_degrees": 10.1},
        ["RF002"],
    ),
    RedFlagCase(
        "RF002 — large 45 ° ROM loss",
        {"pain": 7, "rom_loss_degrees": 45.0},
        ["RF002"],
    ),

    # ── RF003: Mechanical locking ─────────────────────────────────────────────
    RedFlagCase(
        "RF003 — mechanical locking, no other symptoms",
        {"pain": 2, "mechanical_locking": True},
        ["RF003"],
    ),
    RedFlagCase(
        "RF003 — mechanical locking with high pain",
        {"pain": 9, "mechanical_locking": True},
        ["RF003"],
    ),
    RedFlagCase(
        "RF003 — mechanical locking with swelling",
        {"pain": 4, "swelling": "Moderate", "mechanical_locking": True},
        ["RF003"],
    ),

    # ── RF004: Wound change AND fever >38.0 °C ────────────────────────────────
    RedFlagCase(
        "RF004 — wound_change=True with numeric temp 38.5 °C",
        {"pain": 3, "wound_change": True, "temperature_celsius": 38.5},
        ["RF004"],
    ),
    RedFlagCase(
        "RF004 — wound_change=True with boolean fever=True",
        {"pain": 2, "wound_change": True, "fever": True},
        ["RF004"],
    ),
    RedFlagCase(
        "RF004 — wound_drainage backward-compat alias with fever",
        {"pain": 1, "wound_drainage": True, "fever": True},
        ["RF004"],
    ),
    RedFlagCase(
        "RF004 — wound_drainage alias with numeric temp 39.0 °C",
        {"pain": 4, "wound_drainage": True, "temperature_celsius": 39.0},
        ["RF004"],
    ),

    # ── RF005: Pain >7/10 for ≥48 h ──────────────────────────────────────────
    RedFlagCase(
        "RF005 — pain 8/10 for exactly 48 h",
        {"pain": 8, "pain_duration_hours": 48.0},
        ["RF005"],
    ),
    RedFlagCase(
        "RF005 — pain 9/10 for 72 h",
        {"pain": 9, "pain_duration_hours": 72.0},
        ["RF005"],
    ),
    RedFlagCase(
        "RF005 — max pain 10/10 for 96 h",
        {"pain": 10, "pain_duration_hours": 96.0},
        ["RF005"],
    ),
    RedFlagCase(
        "RF005 — pain 8/10 for 48.1 h (just above threshold)",
        {"pain": 8, "pain_duration_hours": 48.1},
        ["RF005"],
    ),

    # ── Multi-rule: RF001 + RF004 fire together ───────────────────────────────
    RedFlagCase(
        "Multi — fever 39 °C with calf pain (RF001) and wound change (RF004)",
        {"pain": 5, "temperature_celsius": 39.0, "calf_pain": True, "wound_change": True},
        ["RF001", "RF004"],
    ),
]


# ── Distractors (30) ───────────────────────────────────────────────────────────
# All should return an empty flags list.

DISTRACTORS: list[RedFlagCase] = [
    # ── RF001 near-misses (8) ─────────────────────────────────────────────────
    RedFlagCase(
        "RF001-D — fever but NO calf symptoms",
        {"pain": 3, "fever": True},
    ),
    RedFlagCase(
        "RF001-D — calf swelling but NO fever",
        {"pain": 2, "calf_swelling": True},
    ),
    RedFlagCase(
        "RF001-D — calf pain but NO fever",
        {"pain": 4, "calf_pain": True},
    ),
    RedFlagCase(
        "RF001-D — temp exactly 38.0 °C (not above threshold) with calf swelling",
        {"pain": 3, "temperature_celsius": 38.0, "calf_swelling": True},
    ),
    RedFlagCase(
        "RF001-D — temp 37.9 °C with calf pain",
        {"pain": 2, "temperature_celsius": 37.9, "calf_pain": True},
    ),
    RedFlagCase(
        "RF001-D — numeric temp 38.0 °C with both calf flags",
        {"pain": 5, "temperature_celsius": 38.0, "calf_swelling": True, "calf_pain": True},
    ),
    RedFlagCase(
        "RF001-D — fever=False, calf_swelling=True",
        {"pain": 1, "fever": False, "calf_swelling": True},
    ),
    RedFlagCase(
        "RF001-D — no calf involvement, no fever",
        {"pain": 4, "temperature_celsius": 37.5},
    ),

    # ── RF002 near-misses (6) ─────────────────────────────────────────────────
    RedFlagCase(
        "RF002-D — rom_loss exactly 10.0 ° (not above threshold)",
        {"pain": 3, "rom_loss_degrees": 10.0},
    ),
    RedFlagCase(
        "RF002-D — rom_loss 9.9 °",
        {"pain": 5, "rom_loss_degrees": 9.9},
    ),
    RedFlagCase(
        "RF002-D — rom_loss 5.0 °",
        {"pain": 2, "rom_loss_degrees": 5.0},
    ),
    RedFlagCase(
        "RF002-D — rom_loss 0 °",
        {"pain": 3, "rom_loss_degrees": 0.0},
    ),
    RedFlagCase(
        "RF002-D — rom_loss None (not provided)",
        {"pain": 4},
    ),
    RedFlagCase(
        "RF002-D — small 1.0 ° loss",
        {"pain": 1, "rom_loss_degrees": 1.0},
    ),

    # ── RF003 near-misses (2) ─────────────────────────────────────────────────
    RedFlagCase(
        "RF003-D — mechanical_locking=False explicitly",
        {"pain": 5, "mechanical_locking": False},
    ),
    RedFlagCase(
        "RF003-D — giving_way=True (retained param, no v0.1 rule)",
        {"pain": 3, "giving_way": True},
    ),

    # ── RF004 near-misses (6) ─────────────────────────────────────────────────
    RedFlagCase(
        "RF004-D — wound_change=True but NO fever",
        {"pain": 2, "wound_change": True},
    ),
    RedFlagCase(
        "RF004-D — fever but NO wound change",
        {"pain": 3, "fever": True},
    ),
    RedFlagCase(
        "RF004-D — wound_change=True, temp exactly 38.0 °C",
        {"pain": 4, "wound_change": True, "temperature_celsius": 38.0},
    ),
    RedFlagCase(
        "RF004-D — wound_drainage=True but NO fever",
        {"pain": 1, "wound_drainage": True},
    ),
    RedFlagCase(
        "RF004-D — wound_drainage=True, temp 37.8 °C",
        {"pain": 2, "wound_drainage": True, "temperature_celsius": 37.8},
    ),
    RedFlagCase(
        "RF004-D — wound_change=False, fever=True (no wound involvement)",
        {"pain": 3, "wound_change": False, "fever": True},
    ),

    # ── RF005 near-misses (6) ─────────────────────────────────────────────────
    RedFlagCase(
        "RF005-D — pain 8/10 for 47.9 h (just below 48 h threshold)",
        {"pain": 8, "pain_duration_hours": 47.9},
    ),
    RedFlagCase(
        "RF005-D — pain exactly 7/10 for 48 h (not above 7)",
        {"pain": 7, "pain_duration_hours": 48.0},
    ),
    RedFlagCase(
        "RF005-D — pain 8/10 for 0 h",
        {"pain": 8, "pain_duration_hours": 0.0},
    ),
    RedFlagCase(
        "RF005-D — pain 9/10 for 24 h",
        {"pain": 9, "pain_duration_hours": 24.0},
    ),
    RedFlagCase(
        "RF005-D — pain 6/10 for 72 h (pain not above 7)",
        {"pain": 6, "pain_duration_hours": 72.0},
    ),
    RedFlagCase(
        "RF005-D — pain 7/10 for 96 h (pain not strictly > 7)",
        {"pain": 7, "pain_duration_hours": 96.0},
    ),

    # ── Fully normal presentations (2) ────────────────────────────────────────
    RedFlagCase(
        "Normal — low pain, no symptoms",
        {"pain": 2},
    ),
    RedFlagCase(
        "Normal — moderate pain, mild swelling, all booleans default",
        {"pain": 5, "swelling": "Mild"},
    ),
]


# ── Parametrize IDs ───────────────────────────────────────────────────────────

def _id(case: RedFlagCase) -> str:
    return case.description


# ── True-positive tests ───────────────────────────────────────────────────────

@pytest.mark.parametrize("case", TRUE_POSITIVES, ids=_id)
def test_true_positive(case: RedFlagCase):
    flags = check_red_flags(**case.kwargs)
    assert flags, f"Expected ≥1 flag for: {case.description}"
    fired_ids = {f.rule_id for f in flags}
    for expected_id in case.expected_rule_ids:
        assert expected_id in fired_ids, (
            f"Rule {expected_id} did not fire for: {case.description}\n"
            f"Fired: {fired_ids}"
        )


# ── Distractor tests ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", DISTRACTORS, ids=_id)
def test_distractor(case: RedFlagCase):
    flags = check_red_flags(**case.kwargs)
    assert not flags, (
        f"Expected no flags for: {case.description}\n"
        f"Got: {[f.rule_id for f in flags]}"
    )


# ── Evidence snapshot integrity ───────────────────────────────────────────────

@pytest.mark.parametrize("case", TRUE_POSITIVES, ids=_id)
def test_evidence_snapshot_present(case: RedFlagCase):
    flags = check_red_flags(**case.kwargs)
    for flag in flags:
        assert isinstance(flag, RedFlag)
        assert flag.evidence_snapshot, (
            f"evidence_snapshot is empty for {flag.rule_id} in: {case.description}"
        )
        assert isinstance(flag.evidence_snapshot, dict)


# ── Severity field ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", TRUE_POSITIVES, ids=_id)
def test_severity_is_urgent(case: RedFlagCase):
    flags = check_red_flags(**case.kwargs)
    for flag in flags:
        assert flag.severity == "URGENT", (
            f"Expected severity URGENT for {flag.rule_id}, got {flag.severity!r}"
        )


# ── str() returns message ─────────────────────────────────────────────────────

@pytest.mark.parametrize("case", TRUE_POSITIVES, ids=_id)
def test_str_returns_message(case: RedFlagCase):
    flags = check_red_flags(**case.kwargs)
    for flag in flags:
        assert str(flag) == flag.message


# ── Aggregate sensitivity assertion ──────────────────────────────────────────

def test_sensitivity_100_percent():
    """All 20 true-positive cases must trigger at least one flag."""
    hits = sum(
        1 for case in TRUE_POSITIVES
        if check_red_flags(**case.kwargs)
    )
    sensitivity = hits / len(TRUE_POSITIVES)
    assert sensitivity == 1.0, (
        f"Sensitivity {sensitivity:.1%} — {len(TRUE_POSITIVES) - hits} "
        f"true-positive case(s) returned no flags"
    )


# ── No false positives on distractors ────────────────────────────────────────

def test_no_false_positives():
    """All 30 distractor cases must return an empty flag list."""
    false_positives = [
        case.description
        for case in DISTRACTORS
        if check_red_flags(**case.kwargs)
    ]
    assert not false_positives, (
        f"{len(false_positives)} distractor(s) incorrectly triggered a flag:\n"
        + "\n".join(f"  • {d}" for d in false_positives)
    )
