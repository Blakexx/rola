import inspect
import os
import pathlib
import sys
from unittest.mock import patch

# Set before torch can initialize CUDA: expandable segments stop fragmentation from
# pushing RESERVED memory past the allocator cap below (the gradcheck lesson).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pytest
import torch

# `measure/` on the path, so the bench PACKAGES import by their own names -- the
# repository's existing convention (`measure/bench_regression.py` inserts the same
# directory and imports `bench.cells`). What still needs it is `bench.cells` and the
# bench drivers; the planner CALIBRATION package no longer does -- it moved to
# `rola/planner/calibration/` (post-merge, 2026-08-02) because an installed wheel has
# no `measure/` sibling and every relocated name raised `ModuleNotFoundError` there.
# The runtime still imports none of it: `rola.planner.schedule` holds lazy call
# proxies, and the subprocess probe in `tests/unit/test_planner.py` asserts
# `import rola` leaves `sys.modules` free of the package.
_BENCHMARKS = pathlib.Path(__file__).resolve().parents[1] / "measure"
if _BENCHMARKS.is_dir() and str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

# `tools/` -- the repository's instruments (the dev config among them); conftest imports them like any other
# caller, not as a package.
_TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
if _TOOLS.is_dir() and str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

# The GPU also drives the WSL display: a test process at the memory cap freezes the
# machine, not just the run. The cap turns an oversized run into a loud OOM instead.
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.70)

try:
    from torch.compiler import is_compiling
except ImportError:
    def is_compiling():
        return False

# The device's torch submodule (`torch.cuda`), resolved without depending on the
# fork's `fla.utils`: `rola` supports NVIDIA CUDA only, and the ratification gate
# is what decides which CUDA devices at that.
device_torch_lib = torch.cuda

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption("--oracle-margins", metavar="FILE", default=None,
                     help="append each per-slot oracle comparison's margins to FILE, one JSON line apiece "
                          "(tests/oracle/tolerances.py)")


def pytest_configure(config):
    from tests.oracle import tolerances

    tolerances.MARGINS_FILE = config.getoption("--oracle-margins")


_ORIGINAL_EMPTY = torch.empty
_ORIGINAL_EMPTY_LIKE = torch.empty_like
_ORIGINAL_NEW_EMPTY = torch.Tensor.new_empty


def _is_called_from_fla():
    """Check if the call is from the `rola` package."""
    frame = inspect.currentframe()
    try:
        # Skip the current frame and go up the call stack
        while frame:
            frame = frame.f_back
            if frame is None:
                break

            if hasattr(frame, 'f_code') and hasattr(frame.f_code, 'co_filename'):
                filename = frame.f_code.co_filename
                # Skip conftest.py frames (where the guarded functions are defined)
                if 'conftest.py' in filename:
                    continue
                # Check if this frame is from a test file
                # Look for 'tests/' or 'test_' in the file path
                if 'tests/' in filename or 'test_' in filename:
                    return False

            module = inspect.getmodule(frame)
            if module and hasattr(module, '__name__'):
                # If call is from the `rola` package, apply guard
                if module.__name__.split('.')[0] == 'rola':
                    return True
    finally:
        del frame
    # Default to not guarding if we can't determine
    return False


def _poison(result):
    """Fill a scratch tensor with NaN. Skip requires_grad leaves: inductor's lazy init (e.g. pad_mm)
    allocates grad-leaf scratch via torch.empty, and an in-place fill_ on a leaf-that-requires-grad
    raises — so with torch.compile defaulted on, poisoning those would break compilation, not catch a
    real uninitialized-read bug. (Those tensors are inductor's, not fla buffers.)"""
    if result.requires_grad:
        return result
    if result.is_floating_point():
        result.fill_(float('nan'))
    elif result.is_complex():
        result.fill_(complex(float('nan'), float('nan')))
    return result


def _guarded_empty(*args, **kwargs):
    """Create a tensor filled with NaN instead of uninitialized values."""
    dtype = kwargs.get('dtype') or torch.get_default_dtype()

    if not (dtype.is_floating_point or dtype.is_complex):
        return _ORIGINAL_EMPTY(*args, **kwargs)

    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_EMPTY(*args, **kwargs)

    return _poison(_ORIGINAL_EMPTY(*args, **kwargs))


def _guarded_empty_like(input, **kwargs):
    """Create a tensor filled with NaN instead of uninitialized values."""
    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_EMPTY_LIKE(input, **kwargs)

    if kwargs.get('dtype') is None:
        kwargs['dtype'] = input.dtype

    dtype = kwargs['dtype']
    if not (dtype.is_floating_point or dtype.is_complex):
        return _ORIGINAL_EMPTY_LIKE(input, **kwargs)

    return _poison(_ORIGINAL_EMPTY_LIKE(input, **kwargs))


def _guarded_new_empty(self, *args, **kwargs):
    """Create a tensor filled with NaN instead of uninitialized values."""
    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_NEW_EMPTY(self, *args, **kwargs)

    if kwargs.get('dtype') is None:
        kwargs['dtype'] = self.dtype

    dtype = kwargs['dtype']
    if not (dtype.is_floating_point or dtype.is_complex):
        return _ORIGINAL_NEW_EMPTY(self, *args, **kwargs)

    return _poison(_ORIGINAL_NEW_EMPTY(self, *args, **kwargs))


#: The tiers whose tests exercise `rola`'s tensor math directly. The unit tier is
#: host-side logic and is deliberately outside the guard. A directory named here that
#: does not exist silently disables the guard, which is what
#: `tests/integration/test_poison_guard_vacuity_probe.py` exists to catch.
_POISON_TIERS = ('tests/oracle/', 'tests/integration/')


@pytest.fixture(scope="function", autouse=True)
def poison_torch_memory(request):
    # Only apply the guard to the tiers that reach rola's tensor math.
    path = str(request.node.fspath)
    if not any(tier in path for tier in _POISON_TIERS):
        yield
        return

    with patch('torch.empty', new=_guarded_empty), \
            patch('torch.empty_like', new=_guarded_empty_like), \
            patch('torch.Tensor.new_empty', new=_guarded_new_empty):
        yield
        if hasattr(device_torch_lib, 'synchronize'):
            device_torch_lib.synchronize()


# -----------------------------------------------------------------------------
# THE GPU LOCK -- the battery takes it itself; a caller never wraps
# `pytest` in an external `flock` again (docs/setup.md, docs/testing.md).
# -----------------------------------------------------------------------------
#
# Which tests need it: `_POISON_TIERS` above is already the tier/marker split
# this repo draws between host-side logic (`tests/unit/`) and tests that reach
# `rola`'s CUDA kernels (`tests/oracle/`, `tests/integration/`) -- reused here
# rather than re-derived, since marker coverage alone (the `cuda` marker) is
# NOT exhaustive: only 6 of 21 files under `tests/oracle/` carry it today, the
# rest rely on the DIRECTORY being GPU work by convention. So a test takes the
# lock if its own tier is one of `_POISON_TIERS`, OR it carries the `cuda`
# marker directly (the `tests/unit/` rows that launch a kernel without living
# in a GPU tier).
#
# The lock itself is SESSION-scoped and lazily acquired: the first GPU test in
# a run pulls `_gpu_lock_held` into existence, which opens `gpu_lock()` once
# and holds it for the rest of the pytest process -- so two concurrent
# batteries on this box serialize against each other (the shared gate), and every
# GPU test after the first in ONE battery pays no repeated lock/unlock cost.
def _touches_gpu(request) -> bool:
    path = str(request.node.fspath)
    if any(tier in path for tier in _POISON_TIERS):
        return True
    return request.node.get_closest_marker("cuda") is not None


@pytest.fixture(scope="session")
def _gpu_lock_held():
    from rola_devtools.locks.gpu import gpu_lock

    #: SHARED mode (LOCKS brief, 2026-08-29): a pytest battery is correctness
    #: work (oracle/integration/cuda-marked unit tests), not a measurement --
    #: it can share the device with another correctness run (up to
    #: `host.gpu_shared_slots` of them) rather than excluding every other
    #: GPU-touching tool, which only measured work (`compare`, `ncu`, the
    #: bench harness) needs.
    with gpu_lock(mode="shared"):
        yield


#: THE HOST-COMPUTE BUDGET, PER XDIST WORKER (LOCKS brief item 1: "a pytest
#: tier that forks workers draws 1 per worker"). `PYTEST_XDIST_WORKER` is set
#: by `pytest-xdist` in every worker process (never in the controller), so a
#: bare `pytest` run (no `-n`) draws nothing here -- only a forked worker
#: competes with compiles/sanitizer runs/clang-tidy passes for the shared pool.
@pytest.fixture(scope="session", autouse=True)
def _host_budget_worker_slot():
    if "PYTEST_XDIST_WORKER" not in os.environ:
        yield
        return
    from rola_devtools.locks import host

    with host.acquire(1, label=f"pytest-worker:{os.environ['PYTEST_XDIST_WORKER']}"):
        yield


@pytest.fixture(autouse=True)
def _take_gpu_lock(request):
    if _touches_gpu(request):
        request.getfixturevalue("_gpu_lock_held")
    yield


# -----------------------------------------------------------------------------
# Building a model, in one place
# -----------------------------------------------------------------------------
#
# The public surface is the reduction theorem: a feature map, and linear attention.
# Assembling one is three lines -- producers, the producer, the layer -- and every test
# that needs a model needs the same three, so they live here rather than being spelled
# out per file. A test that is ABOUT construction (validation, refusal, the expert pin)
# calls the real constructors directly; these are for the tests that need a model to
# have something to run.


@pytest.fixture
def dev_config_env(tmp_path):
    """`make(host={...}, ...)` writes those sections to a fresh dev config directory (`tools/dev_config.py`) and returns
    an environment whose children read it, the in-process loader's cache cleared."""
    import json

    import dev_config

    def make(**sections) -> dict:
        root = tmp_path / "dev_config"
        root.mkdir(exist_ok=True)
        for name, values in sections.items():
            (root / f"{name}.json").write_text(json.dumps(values))
        env = dict(os.environ)
        env[dev_config.POINTER] = str(root)
        dev_config.reload()
        return env

    yield make
    dev_config.reload()


def build_routes(*, hidden_size=64, num_heads=4, widths=(16,), routing=None,
                 gain=True, bias=False,
                 gain_bias_init=None, gain_config=None):
    """One :class:`rola.RouteProducer` covering every level in ``widths``."""
    from rola import dense_routing
    from rola.routing.producer import RouteProducer

    levels = [dense_routing(w) if routing is None else routing.at(w) for w in widths]
    return RouteProducer(
        levels, hidden_size=hidden_size, num_heads=num_heads, bias=bias,
        gain=gain, gain_bias_init=gain_bias_init,
        gain_config=gain_config)


def build_layer(*, hidden_size=64, num_heads=4, d_v=32, widths=(16,), routing=None,
                decay=None, decay_rate=2.0 ** -8, gain=True,
                bias=False, expert=None, layer_idx=None, **kwargs):
    """A :class:`rola.RoLA` over :func:`build_routes`.

    ``decay`` takes ``None`` (no decay), ``'learned'`` or ``'constant'`` -- the two
    shipped decay SOURCES -- rather than a rate, because which source a model uses is
    the choice the slot exists to express.
    """
    from rola import ConstantDecay, LearnedDecay, RoLA

    routes = build_routes(hidden_size=hidden_size, num_heads=num_heads,
                          widths=widths, routing=routing,
                          gain=gain, bias=bias, **kwargs)
    sources = {"learned": LearnedDecay, "constant": ConstantDecay}
    source = None if decay is None else sources[decay](
        decay_rate, widths=routes.widths, num_heads=num_heads)
    return RoLA(routes, d_v=d_v, decay=source,
                layer_idx=layer_idx, expert=expert)


# -----------------------------------------------------------------------------
# Driving the producer from bare parameter tensors
# -----------------------------------------------------------------------------
#
# The low-level producer gates in tests/oracle/test_producer.py and
# tests/integration/test_production_levels.py drive raw parameter tensors directly
# (route_W, gain_W, ... as bare leaves) rather than a module's parameters, which
# is what lets them isolate a backward-through-every-tensor gate. The production
# orchestration is `RouteProducer` -- an nn.Module over nn.Parameters -- so
# `produce_routing` builds one and runs it under
# `torch.nn.utils.stateless.functional_call` with the caller's tensors standing
# in for its parameters. The graph reaches the caller's leaves and the numerics
# are the REAL producer's, not a second derivation of them.


def resolved_routing_from_topology(topology):
    """A `ResolvedRouting` over a `Topology`'s own levels -- exactly what
    `RouteProducer.__init__` builds for itself
    (`self.routing = ResolvedRouting(levels=self.levels)`)."""
    from rola.routing.types import ResolvedRouting

    return ResolvedRouting(levels=topology.levels)


def fuse_gain_columns(route_W, gain_W):
    """`route_W` with `gain_W` as its trailing columns -- THE fused parameter's layout.

    THE GAIN'S WEIGHT IS `route_W`'s TRAILING SPAN. `gain_W` arrives in its own
    `[H, sides, hidden]` layout because that is what these gates hold; defined once here
    so a gate that needs the pre-fold gain builds the same tensor the producer runs on.
    """
    return torch.cat([route_W, gain_W.transpose(-1, -2)], dim=-1)


def raw_side_gain(x, route_W, gain_W, gain_bias, resolved):
    """`g_write` BEFORE the mass fold, through the library's own two pieces.

    Not a second implementation: the projection is `_packed_router_logits`' gain span
    and the nonlinearity is `SideGain`, which is exactly what the producer composes --
    what the producer does NOT expose is the value between them, which is what a fold
    gate has to see.
    """
    from rola.routing.projection import _packed_router_logits
    from rola.routing.side_gain import GainConfig, SideGain

    columns = gain_W.shape[1]
    _, _, gain_logits = _packed_router_logits(
        x, fuse_gain_columns(route_W, gain_W), resolved, route_bias=None, h_w=None,
        projection_dtype=x.dtype, gain_columns=columns)
    gain = SideGain(num_heads=gain_W.shape[0], config=GainConfig.from_public(bool(columns)))
    with torch.no_grad():
        gain.gain_bias.copy_(gain_bias)
    return gain(gain_logits, batch=x.shape[0], tokens=x.shape[1], dtype=x.dtype,
                device=x.device)


def produce_routing(x, x_w, v, route_W, route_bias, gain_W, gain_bias, topology):
    """The producer, run over bare parameter tensors -> a plain namespace.

    ``v`` is accepted (matching the call sites) and unused: callers need it only
    afterwards, to run the oracle. The return carries the field names those gates
    read (`read_levels`/`write_levels`/`g_write`). There is no read-side gain: the
    readout is a ratio, which fixes it to 1 identically.
    """
    from types import SimpleNamespace

    from torch.func import functional_call

    from rola.routing.producer import RouteProducer

    H, hidden_size, _ = route_W.shape
    producer = RouteProducer(
        topology.levels, hidden_size=hidden_size, num_heads=H,
        bias=route_bias is not None)
    parameters = {"route_W": fuse_gain_columns(route_W, gain_W),
                  "side_gain.gain_bias": gain_bias}
    if route_bias is not None:
        parameters["route_bias"] = torch.cat(
            [route_bias, route_bias.new_zeros(H, gain_W.shape[1])], dim=-1)
    streams = (x,) if x_w is None else (x, x_w)
    routes = functional_call(producer, parameters, streams)

    return SimpleNamespace(
        read_levels=routes.read_simplex(), write_levels=routes.write,
        g_write=routes.g_write)
