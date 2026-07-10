"""`TurnArbiter` — exclusive ownership of "whose turn is it" across every
turn source. See core/turn_arbiter.py's module docstring for the full
rationale (it wraps, not replaces, `Assistant._turn_lock`).
"""

from core.turn_arbiter import TurnArbiter


def test_uncontended_acquire_and_release():
    arb = TurnArbiter()
    assert arb.owner is None
    assert arb.try_acquire("sat-a") is True
    assert arb.owner == "sat-a"
    arb.release("sat-a")
    assert arb.owner is None


def test_contended_second_try_acquire_fails_while_held():
    arb = TurnArbiter()
    assert arb.try_acquire("sat-a") is True
    assert arb.try_acquire("sat-b") is False
    # The holder is unaffected by the loser's failed attempt.
    assert arb.owner == "sat-a"


def test_release_frees_it_for_the_next_acquire():
    arb = TurnArbiter()
    arb.try_acquire("sat-a")
    arb.release("sat-a")
    assert arb.try_acquire("sat-b") is True
    assert arb.owner == "sat-b"


def test_double_release_is_a_noop():
    arb = TurnArbiter()
    arb.try_acquire("sat-a")
    arb.release("sat-a")
    arb.release("sat-a")  # must not raise (e.g. RuntimeError: unlocked lock)
    assert arb.owner is None
    # And the arbiter is still genuinely usable afterward.
    assert arb.try_acquire("sat-b") is True


def test_release_by_non_owner_does_not_release_the_real_owner():
    arb = TurnArbiter()
    arb.try_acquire("sat-a")
    arb.release("sat-b")  # sat-b never held it
    assert arb.owner == "sat-a"
    # The real owner's turn is untouched — a third party can't steal it.
    assert arb.try_acquire("sat-c") is False


def test_release_of_never_acquired_arbiter_is_a_noop():
    arb = TurnArbiter()
    arb.release("sat-a")  # never acquired anything
    assert arb.owner is None
