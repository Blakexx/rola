"""THE FIXTURE'S OWN TEETH. Every check here is run against a PLANTED violation.

`stream_discipline.py` is a fixture other tests instantiate to gate a stream-crossing
path. A fixture that cannot be shown failing is worse than none: it converts "we
checked" into a sentence nobody can falsify. So each helper is exercised twice --
against a correct path, which must pass, and against a deliberately broken one, which
must fail.

The synthetic paths here are the smallest programs that have the defect: a fork with
no join, a tensor crossing a stream without declaring its lifetime, a publication read
before it is joined. They are not the paged path and are not meant to be; they are
what proves the instrument reads.
"""
from __future__ import annotations

import pytest
import torch

from tests.integration.stream_discipline import (
    OrderingSpy,
    allocator_pressure,
    assert_capture_rejoins,
    assert_recorded_on,
    capture_refusal,
    record_stream_spy,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="stream discipline is a CUDA property"),
]

#: A fork/join around one elementwise op -- the smallest region that HAS a join to
#: omit. The warm-up is required by the allocator before any capture and is part of
#: the recipe, not part of the subject.
_CAPTURE_BODY = """
    a = torch.ones(1024, device="cuda")
    side = torch.cuda.Stream()
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            (a * 2).sum()
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            b = a * 2
        {join}
        b.sum()
    print("OK")
"""


def test_an_unjoined_fork_refuses_to_close_a_capture():
    """PLANTED VIOLATION: the join is deleted. `cudaStreamEndCapture` must refuse.

    This is the mechanical detector working on THIS driver, which is the only reason
    to trust it anywhere else in the suite.
    """
    refusal = capture_refusal(_CAPTURE_BODY.format(join=""))
    assert "unjoined work" in refusal, (
        f"an unjoined fork CLOSED the capture, so the detector is asleep on this "
        f"driver and every assertion built on it is vacuous:\n{refusal[-2000:]}")


def test_the_same_region_with_its_join_captures_cleanly():
    """The other half: the refusal above must be about the missing join and not
    about the region being uncapturable for some unrelated reason."""
    assert_capture_rejoins(
        _CAPTURE_BODY.format(join="torch.cuda.current_stream().wait_stream(side)"))


# ---------------------------------------------------------------------------
# The ordering spy
# ---------------------------------------------------------------------------

class _Publisher:
    """A publication with a join, and a switch that removes the join."""

    def __init__(self, join: bool):
        self.join = join
        self.stream = torch.cuda.Stream()
        self.ready = None
        self.buffer = torch.zeros(1024, device="cuda")

    def publish(self):
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.buffer.fill_(3.0)
        self.ready = torch.cuda.Event()
        self.ready.record(self.stream)

    def wait(self):
        if self.join and self.ready is not None:
            torch.cuda.current_stream().wait_event(self.ready)

    def consume(self):
        return self.buffer.sum()


def _observe(monkeypatch, publisher):
    spy = OrderingSpy()
    spy.watch(monkeypatch, _Publisher, "publish", label="publish")
    spy.watch(monkeypatch, _Publisher, "wait", label="wait")
    spy.watch(monkeypatch, _Publisher, "consume", label="consume")
    publisher.publish()
    publisher.wait()
    publisher.consume()
    return spy


def test_the_ordering_spy_accepts_a_joined_publication(monkeypatch):
    spy = _observe(monkeypatch, _Publisher(join=True))
    spy.assert_between("publish", "wait", "consume")


def test_the_ordering_spy_rejects_a_publication_read_without_its_join(monkeypatch):
    """PLANTED VIOLATION: `wait()` is called but joins nothing -- which is exactly
    the shape of the escaped bug, where the call site omitted the one join and every
    value was still correct until the timing moved."""
    publisher = _Publisher(join=False)
    spy = OrderingSpy()
    spy.watch(monkeypatch, _Publisher, "publish", label="publish")
    spy.watch(monkeypatch, _Publisher, "consume", label="consume")
    publisher.publish()
    publisher.consume()
    with pytest.raises(AssertionError, match="without being joined"):
        spy.assert_between("publish", "wait", "consume")


def test_an_inner_join_cannot_satisfy_the_outer_claim(monkeypatch):
    """A join performed INSIDE the publication orders that publication against the
    previous one; it is not the join before the consumer. The spy must record the two
    distinctly or the outer claim passes on the inner call."""
    spy = OrderingSpy()

    class _Nested:
        def outer(self):
            self.inner()

        def inner(self):
            return None

        def consume(self):
            return None

    spy.watch(monkeypatch, _Nested, "outer", label="publish")
    spy.watch(monkeypatch, _Nested, "inner", label="wait", nested_label="wait_inner")
    spy.watch(monkeypatch, _Nested, "consume", label="consume")
    obj = _Nested()
    obj.outer()
    obj.consume()
    assert spy.events == ["publish", "wait_inner", "consume"], spy.events
    with pytest.raises(AssertionError, match="without being joined"):
        spy.assert_between("publish", "wait", "consume")


# ---------------------------------------------------------------------------
# The lifetime spy
# ---------------------------------------------------------------------------

def test_the_lifetime_spy_sees_a_declared_crossing_and_misses_an_undeclared_one(
        monkeypatch):
    """Both directions, because either alone is consistent with the spy being blind."""
    side = torch.cuda.Stream()
    declared = torch.arange(64, device="cuda")
    undeclared = torch.arange(64, device="cuda")

    with record_stream_spy(monkeypatch) as recorded:
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            declared.record_stream(side)
            declared.sum()
            undeclared.sum()
        torch.cuda.current_stream().wait_stream(side)

    assert_recorded_on(recorded, [id(declared)], side, what="the declared tensor")
    with pytest.raises(AssertionError, match="WITHOUT record_stream"):
        assert_recorded_on(recorded, [id(undeclared)], side,
                           what="the undeclared tensor")


# ---------------------------------------------------------------------------
# The pressure burst
# ---------------------------------------------------------------------------

def test_the_pressure_burst_actually_recycles_allocator_blocks():
    """NON-VACUITY OF THE WINDOW. A burst that allocates nothing opens no window,
    and every cell that runs under it then proves only that nothing happened."""
    torch.cuda.empty_cache()
    before = torch.cuda.memory_stats()["segment.all.allocated"]
    allocator_pressure(rounds=16)
    after = torch.cuda.memory_stats()["segment.all.allocated"]
    assert after > before or torch.cuda.memory_stats()["allocation.all.allocated"] > 0, (
        "the burst allocated nothing; the window a lifetime bug needs was never open")
