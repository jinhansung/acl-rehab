"""Sophia — 17-year-old basketball player, week 16, cleared for return-to-sport testing."""
import pytest
from agent.red_flags import check_red_flags


def test_sophia_rts_day_clear():
    flags = check_red_flags(pain=1, swelling="None", giving_way=False)
    assert not flags


def test_sophia_fever_post_workout():
    flags = check_red_flags(pain=2, swelling="Mild", giving_way=False, fever=True)
    assert any("fever" in f.lower() for f in flags)


def test_sophia_pain_during_agility():
    flags = check_red_flags(pain=8, swelling="None", giving_way=False)
    assert any("pain" in f.lower() for f in flags)
