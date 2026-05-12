"""Maya — 22-year-old soccer player, week 2 post-op, high motivation, no red flags."""
import pytest
from agent.red_flags import check_red_flags


def test_maya_day1_check_in_clear():
    flags = check_red_flags(pain=3, swelling="Mild", giving_way=False)
    assert not flags


def test_maya_day3_pain_spike_flagged():
    flags = check_red_flags(pain=8, swelling="Moderate", giving_way=False)
    assert any("pain" in f.lower() for f in flags)


def test_maya_day5_back_to_baseline():
    flags = check_red_flags(pain=2, swelling="None", giving_way=False)
    assert not flags
