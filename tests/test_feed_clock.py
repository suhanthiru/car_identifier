"""FeedClock: the replay pause.

The property that matters is not "it stops" but "it stops WITHOUT
distorting the timeline". Every camera runs as its own task off one shared
clock, and the reasoning layer scores cross-camera gaps against observed
transit windows -- so if a pause let those gaps drift, a paused demo would
quietly produce different verdicts than an unpaused one.
"""
import asyncio

import pytest

from server.feed import FeedClock


class _State:
    def __init__(self, paused: bool = False):
        self.feed_paused = paused


def test_elapsed_advances_when_running():
    async def go():
        clock = FeedClock(_State(paused=False))
        await asyncio.sleep(0.05)
        return clock.elapsed()

    assert asyncio.run(go()) >= 0.04


def test_paused_time_is_not_counted():
    """The whole point: wall time spent paused must not advance the replay."""
    async def go():
        state = _State(paused=False)
        clock = FeedClock(state)
        await asyncio.sleep(0.05)
        state.feed_paused = True
        clock._sync()                 # observe the pause, as wait_until does
        before = clock.elapsed()
        await asyncio.sleep(0.15)     # a long stare at a frozen console
        during = clock.elapsed()
        state.feed_paused = False
        clock._sync()
        after = clock.elapsed()
        return before, during, after

    before, during, after = asyncio.run(go())
    assert during == pytest.approx(before, abs=0.02), "clock ran while paused"
    assert after == pytest.approx(before, abs=0.02), "pause leaked into the timeline"


def test_wait_until_returns_immediately_when_already_due():
    async def go():
        clock = FeedClock(_State(paused=False))
        await asyncio.sleep(0.05)
        start = asyncio.get_running_loop().time()
        await clock.wait_until(0.0)   # long past
        return asyncio.get_running_loop().time() - start

    assert asyncio.run(go()) < 0.05


def test_wait_until_blocks_while_paused_then_proceeds():
    """A task waiting on a due time must not fire until the clock resumes."""
    async def go():
        state = _State(paused=True)
        clock = FeedClock(state)
        fired = asyncio.Event()

        async def waiter():
            await clock.wait_until(0.01)
            fired.set()

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.25)
        still_waiting = not fired.is_set()
        state.feed_paused = False
        await asyncio.wait_for(task, timeout=2.0)
        return still_waiting, fired.is_set()

    still_waiting, fired_after = asyncio.run(go())
    assert still_waiting, "wait_until fired while the feed was paused"
    assert fired_after, "wait_until never fired after resume"


def test_relative_spacing_survives_a_pause():
    """Two events 0.10s apart must stay 0.10s apart across a pause -- this is
    what keeps cross-camera transit times honest."""
    async def go():
        state = _State(paused=False)
        clock = FeedClock(state)
        await clock.wait_until(0.05)
        first = clock.elapsed()
        state.feed_paused = True
        clock._sync()
        await asyncio.sleep(0.2)
        state.feed_paused = False
        clock._sync()
        await clock.wait_until(0.15)
        return clock.elapsed() - first

    assert asyncio.run(go()) == pytest.approx(0.10, abs=0.05)


def test_missing_state_means_never_paused():
    """A feed run without a server (tests, scripts) must not need the flag."""
    async def go():
        clock = FeedClock(None)
        await clock.wait_until(0.0)
        return clock.paused

    assert asyncio.run(go()) is False
