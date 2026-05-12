"""Helen — 45-year-old recreational hiker, week 8, cautious, moderate swelling history."""
import pytest
from agent.red_flags import check_red_flags


def test_helen_typical_day_clear():
    flags = check_red_flags(pain=4, swelling="Mild", giving_way=False)
    assert not flags


def test_helen_giving_way_episode():
    flags = check_red_flags(pain=5, swelling="Mild", giving_way=True)
    assert any("instability" in f.lower() for f in flags)


def test_helen_severe_swelling_after_hike():
    flags = check_red_flags(pain=5, swelling="Severe", giving_way=False)
    assert any("swelling" in f.lower() for f in flags)
