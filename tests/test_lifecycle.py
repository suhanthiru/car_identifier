"""State machine tests for the track lifecycle."""
from tracking.lifecycle import (
    COAST_AFTER_S, CONFIRMED, COASTING, LOST, LOST_AFTER_S, TENTATIVE,
    Track, on_association, on_rejection, on_tick,
)


def test_starts_tentative():
    assert Track("tgt-1").state == TENTATIVE


def test_plate_grade_promotes_immediately():
    t = on_association(Track("tgt-1"), 10.0, plate_grade=True)
    assert t.state == CONFIRMED


def test_two_appearance_sightings_promote():
    t = on_association(Track("tgt-1"), 10.0, plate_grade=False)
    assert t.state == TENTATIVE
    t = on_association(t, 60.0, plate_grade=False)
    assert t.state == CONFIRMED


def test_rejection_breaks_streak():
    t = on_association(Track("tgt-1"), 10.0, plate_grade=False)
    t = on_rejection(t, 20.0)
    t = on_association(t, 30.0, plate_grade=False)
    assert t.state == TENTATIVE, "streak must restart after a rejection"


def test_confirmed_coasts_then_loses():
    t = on_association(Track("tgt-1"), 100.0, plate_grade=True)
    t = on_tick(t, 100.0 + COAST_AFTER_S + 1)
    assert t.state == COASTING
    t = on_tick(t, 100.0 + LOST_AFTER_S + 1)
    assert t.state == LOST


def test_coasting_boundary_is_exclusive():
    t = on_association(Track("tgt-1"), 100.0, plate_grade=True)
    assert on_tick(t, 100.0 + COAST_AFTER_S).state == CONFIRMED
    assert on_tick(t, 100.0 + COAST_AFTER_S + 0.1).state == COASTING


def test_reacquisition_from_coasting_is_tentative():
    t = on_association(Track("tgt-1"), 100.0, plate_grade=True)
    t = on_tick(t, 100.0 + COAST_AFTER_S + 1)
    t = on_association(t, 500.0, plate_grade=False)
    assert t.state == TENTATIVE
    assert t.consecutive_sightings == 1


def test_reacquisition_with_plate_confirms_instantly():
    t = on_association(Track("tgt-1"), 100.0, plate_grade=True)
    t = on_tick(t, 100.0 + LOST_AFTER_S + 500)
    t = on_tick(t, 100.0 + LOST_AFTER_S + 900)
    assert t.state in (COASTING, LOST)
    t = on_association(t, 2000.0, plate_grade=True)
    assert t.state == CONFIRMED


def test_tentative_track_eventually_lost():
    t = on_association(Track("tgt-1"), 100.0, plate_grade=False)
    t = on_tick(t, 100.0 + LOST_AFTER_S + 1)
    assert t.state == LOST
