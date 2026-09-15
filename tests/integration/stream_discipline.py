"""THE STREAM-DISCIPLINE FIXTURE: what every new stream-crossing path must instantiate.

Two bugs escaped this repository and both were host-side Python stream ordering: a
`PageArena.wait()` never called before the consumer launch, and `atom_ids` crossing to
a side stream with no `record_stream`. Neither is a device memory hazard, so neither
is visible to `compute-sanitizer` -- `racecheck` detects on-chip SHARED MEMORY hazards
only, and there is no runtime detector anywhere for a forgotten cross-stream
dependency on already-allocated memory. That gap is real and confirmed.

What caught them is in this file. It was four bespoke cells guarding one module; it is
now a fixture, because the next path that crosses a stream needs the same four checks
and copying them is how three of the four end up missing.

| check | what it makes deterministic |
|---|---|
| :class:`OrderingSpy` | the publication is JOINED before the consumer reads it |
| :func:`record_stream_spy` | a tensor crossing to another stream declares its lifetime |
| :func:`allocator_pressure` | the window a lifetime bug needs is actually opened |
| :func:`assert_agrees_under_launch_blocking` | async and serialized runs agree bit-for-bit |
| :func:`assert_capture_rejoins` | a fork that never rejoins REFUSES to close a CUDA graph capture |

The last is the only MECHANICAL detector of the missing-join class that exists, and it
is not ours: `cudaStreamEndCapture` returns `cudaErrorStreamCaptureUnjoined` when a
stream forked inside the captured region never joins back to the origin. Measured on
this box (sm_86, CUDA 12.4): a joined fork closes the capture, an unjoined one raises
`CUDA error: capturing stream has unjoined work`. Its LIMITS are equally measured and
are stated at :func:`assert_capture_rejoins` -- it can only see a region that is
capture-legal, and the production paged path is not one.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from contextlib import contextmanager

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OrderingSpy:
    """Record a named sequence of calls, then assert the order they must occur in.

    A path that publishes on a side stream and then reads the publication has an
    ORDER its correctness depends on, and that order is invisible to any assertion
    about values -- the values are right until the timing changes. Recording the
    sequence turns it into a deterministic test.
    """

    def __init__(self):
        self.events: list[str] = []
        self._depth: dict[str, int] = {}

    def watch(self, monkeypatch, owner, name, *, label=None, nested_label=None):
        """Record every call to `owner.name`.

        `nested_label` names calls that happen INSIDE another watched call: a join
        performed by a reset, for instance, orders that reset against the previous
        publication and cannot stand in for the join before the consumer. Recording
        the two distinctly is what stops the inner one satisfying the outer claim.
        """
        label = label or name
        original = getattr(owner, name)

        def spy(*args, **kwargs):
            inner = nested_label and any(v for k, v in self._depth.items() if k != label)
            self.events.append(nested_label if inner else label)
            self._depth[label] = self._depth.get(label, 0) + 1
            try:
                return original(*args, **kwargs)
            finally:
                self._depth[label] -= 1

        monkeypatch.setattr(owner, name, spy)
        return self

    def assert_between(self, first: str, join: str, last: str) -> None:
        assert first in self.events and last in self.events, self.events
        i, j = self.events.index(first), self.events.index(last)
        assert j > i, f"{last!r} did not follow {first!r}: {self.events}"
        assert any(e == join for e in self.events[i + 1:j]), (
            f"no {join!r} between {first!r} and {last!r}: the publication is read "
            f"without being joined, so the reader sees it only when the timing "
            f"happens to allow (observed order: {self.events})")


@contextmanager
def record_stream_spy(monkeypatch):
    """Collect `(id(tensor), stream)` for every `record_stream` call in the block.

    A tensor handed to another stream and then freed on this one is bytes the caching
    allocator may reuse while the other stream still reads them. `record_stream` is
    the declaration that stops that, there is no debug mode anywhere that flags its
    absence, so the call itself is what gets observed.
    """
    recorded: list[tuple[int, object]] = []
    original = torch.Tensor.record_stream

    def spy(self, stream):
        recorded.append((id(self), stream))
        return original(self, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", spy)
    yield recorded


def assert_recorded_on(recorded, tensor_ids, stream, *, what: str) -> None:
    hits = [tid for tid, got in recorded
            if tid in set(tensor_ids)
            and getattr(got, "cuda_stream", None) == stream.cuda_stream]
    assert hits, (
        f"{what} crossed to another stream WITHOUT record_stream on it: the caching "
        f"allocator may hand its bytes to another consumer while that stream is "
        f"still reading them")


def allocator_pressure(rounds: int = 64, device: str = "cuda") -> None:
    """Alloc/free churn sized to make the caching allocator recycle blocks fast.

    Opened in the window between the publication and the read, this is the condition
    a lifetime bug needs. A cell that never opens the window proves the path is safe
    only when nothing else is happening, which is not the claim.
    """
    for _ in range(rounds):
        block = torch.empty(1 << 20, device=device)
        del block
        torch.randn(1 << 12, device=device).sum()


def assert_agrees_under_launch_blocking(body: str, *, timeout: int = 600) -> None:
    """Run `body` in a `CUDA_LAUNCH_BLOCKING=1` child and require it to succeed.

    Blocking serializes launches, which collapses the window a missing join needs. It
    REPORTS nothing by itself; it makes the async and serialized runs disagree, and
    the comparison is what turns that into a signal. `body` therefore asserts its own
    bit-identity and prints OK.
    """
    child = f"import sys; sys.path.insert(0, {ROOT!r})\n" + textwrap.dedent(body)
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
    proc = subprocess.run([sys.executable, "-c", child], capture_output=True,
                          text=True, env=env, timeout=timeout)
    assert proc.returncode == 0 and proc.stdout.strip().endswith("OK"), (
        f"the CUDA_LAUNCH_BLOCKING=1 child disagreed with the async run -- that "
        f"disagreement IS the stream-race signature:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")


#: The capture error text this driver produces for an unjoined fork. Matched rather
#: than the numeric `cudaErrorStreamCaptureUnjoined`, which torch does not surface.
_UNJOINED = "unjoined work"


def _capture_child(body: str) -> subprocess.CompletedProcess:
    """Run one capture attempt in its OWN process.

    A failed capture leaves the context in an error state that later CUDA calls
    inherit, so every attempt is isolated -- otherwise one deliberate failure would
    poison the rest of the session and the poisoning would look like a second bug.
    """
    child = (f"import sys; sys.path.insert(0, {ROOT!r})\n"
             "import torch\n"
             "torch.cuda.set_per_process_memory_fraction(0.65)\n"
             + textwrap.dedent(body))
    return subprocess.run([sys.executable, "-c", child], capture_output=True,
                          text=True, timeout=600)


def assert_capture_rejoins(body: str) -> None:
    """A fork inside `body` must rejoin the origin stream, or the capture refuses.

    THE ONLY MECHANICAL DETECTOR OF A MISSING JOIN. `cudaStreamEndCapture` returns
    `cudaErrorStreamCaptureUnjoined` when a stream forked inside the captured region
    never joins back, so wrapping a stream-crossing path in a throwaway capture turns
    a timing-dependent context poisoning into a hard error at a known line.

    **LIMITS, measured rather than assumed.** It validates only the captured region;
    the path must be CAPTURE-LEGAL; and it sees reachability back to the origin
    stream and nothing else -- an event recorded or waited on the WRONG stream is
    invisible to it. Measured on this box: the production paged path is not
    capture-legal, because `PageArena` owns a persistent side stream and records its
    readiness event OUTSIDE any capture, which the capture rejects as a
    cross-capture dependency before the body runs. So this is the contract for paths
    that CAN be captured, and `test_stream_discipline_fixture.py` proves the detector
    itself works on this driver in both directions.

    `body` must print OK on success.
    """
    proc = _capture_child(body)
    assert proc.returncode == 0 and proc.stdout.strip().endswith("OK"), (
        f"the region did not close a CUDA graph capture. If the message names "
        f"{_UNJOINED!r}, a stream forked inside it never rejoined the origin -- the "
        f"exact defect class that escaped this repository twice:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")


def capture_refusal(body: str) -> str:
    """The refusal text a capture attempt produced, for a test that WANTS one."""
    proc = _capture_child(body)
    return proc.stdout + proc.stderr
