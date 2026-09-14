# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The public op stack (`rola/interface.py`) and the O-3 name contract.

Two things are gated here. The first is THE ENVELOPE -- what `rola_op` runs and what
it refuses -- because the layer-dispatch autograd gap is a failure that produces no
exception and no wrong number, only a `grad_fn` that quietly is not there, and the
answer to it is a refusal rather than a second arm. The second is the NAME contract:
no public spelling says `v3`.

The CUDA arm's own numerics are gated in `tests/oracle/`; nothing here
re-measures them.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch
from conftest import build_routes

import rola
from rola.expert import envelope_refusal, require_envelope
from rola.interface import operands_of, requires_backward

# --- the predicate -----------------------------------------------------------


def test_the_predicate_is_false_whenever_grad_is_globally_off():
    leaf = torch.zeros(2, requires_grad=True)
    assert requires_backward(leaf) is True
    with torch.no_grad():
        assert requires_backward(leaf) is False


def test_the_predicate_walks_into_the_per_level_tuples():
    """The routing allocations arrive as one tensor per level. A predicate that only
    looked at its top-level arguments would see a tuple, find no `requires_grad`, and
    send a grad-requiring call to a forward-only kernel."""
    leaf = torch.zeros(2, requires_grad=True)
    assert requires_backward((torch.zeros(2), leaf)) is True
    assert requires_backward([[torch.zeros(2), (leaf,)]]) is True
    assert requires_backward((torch.zeros(2), torch.zeros(2))) is False


def test_the_predicate_ignores_non_tensors_rather_than_guessing():
    assert requires_backward(None, "plan", 3, object()) is False


def test_the_layer_and_the_op_ask_the_same_question():
    """One predicate, or the two surfaces drift into two slightly different answers.
    Asserted structurally -- the layer imports it -- not by comparing behaviours."""
    import rola.layer as layer_module

    assert layer_module.requires_backward is requires_backward
    assert layer_module.operands_of is operands_of


def _dense_routing():
    from rola.routing.types import TiedRouting

    return TiedRouting(width=1, op=rola.SoftmaxActivation())


def _tiny_routing(requires_grad: bool, device: str = "cpu"):
    """A real `RouteFactors` bundle, on the smallest topology the tests need.

    Old: five loose tensors (one routing level, its gains) plus a `_StubPlan`
    carrying only `topology`/`normalization`. New: `rola_op` takes a
    `RouteFactors`, obtainable only from `RouteProducer`, so the fixture builds
    one -- `requires_grad` on the source stream `x` reaches every tensor the
    producer derives (read/write levels, gains), exactly as `requires_grad` on the
    five loose tensors used to.
    """
    torch.manual_seed(3)
    B, T, H, d_v, hidden, w = 1, 8, 1, 16, 8, 4
    producer = build_routes(
        hidden_size=hidden, num_heads=H, widths=(w,), routing=_dense_routing()
    ).to(device)
    x = torch.randn(B, T, hidden, device=device, requires_grad=requires_grad)
    routes = producer(x)
    v = torch.randn(B, T, H, d_v, device=device, requires_grad=requires_grad)
    return v, routes


def test_the_prefill_arm_refuses_by_naming_the_ruling():
    """K31: the prefill execution arm is DELETED, and the op says so by name --
    the ruling, the queue card and the baseline tag -- before it looks at the
    bundle at all. The refusal is the surface until R2 lands (the layer-dispatch
    autograd-gap lesson: a deleted arm must raise, never sever silently)."""
    v, routes = _tiny_routing(requires_grad=True)
    assert requires_backward(v, *routes.tensors()) is True
    with pytest.raises(NotImplementedError, match="the prefill arm is deleted"):
        rola.rola_op(routes, v)
    with pytest.raises(NotImplementedError, match="baseline/pre-k31"):
        rola.rola_op(routes, v)


def test_rola_op_has_no_consumer_only_kwargs_to_silently_ignore():
    """OLD: `rola_func` accepted CUDA-consumer-only options (`paging`, ...) on
    BOTH arms and refused them explicitly with `ValueError` on the reference
    arm -- one function, two arms, one of which had to actively reject what the
    other used. `rola_op` structurally cannot have that failure mode: paging,
    scheduling and every other CUDA-consumer detail live behind
    `rola.expert.execute`'s own seam, reachable only from the CUDA path, and
    `rola_op`'s signature carries no CUDA-consumer-only parameter at all -- so
    passing one is a plain `TypeError` (unknown keyword) regardless of which arm
    the call would otherwise take, rather than a case the op has to notice and
    refuse."""
    v, routes = _tiny_routing(requires_grad=True)
    with pytest.raises(TypeError):
        rola.rola_op(routes, v, paging=object())


def test_an_explicit_no_grad_context_is_how_an_inference_site_calls_the_op():
    """OLD: `rola_forward` was a separate entrypoint that established its own
    no-grad context internally, so an inference caller could not be wrong about
    it by forgetting to wrap the call. `rola_op` merges the two entrypoints into
    one: there is no separate inference spelling any more, and the caller's own
    `torch.no_grad()` is what an inference site now uses -- exactly as any other
    PyTorch op. The surviving half of the old claim: wrapped in `torch.no_grad()`,
    an every-input-requires-grad call is ADMITTED past the gradient clause, because
    the predicate is gradient REACHABILITY and not the `requires_grad` flag.
    """
    v, routes = _tiny_routing(requires_grad=True)
    with torch.no_grad():
        assert requires_backward(v, *routes.tensors()) is False
        #: CPU, so the DEVICE clause is what speaks -- the gradient clause did not.
        assert "CUDA kernel" in envelope_refusal(routes, v)


def test_the_refusal_is_the_same_sentence_with_and_without_a_gradient():
    """The K31 refusal speaks whether or not a gradient is attached: the same
    sentence must come back both times, so a caller cannot be told two different
    things about one bundle."""
    v, routes = _tiny_routing(requires_grad=True)
    with pytest.raises(NotImplementedError, match="the prefill arm is deleted"):
        rola.rola_op(routes, v)
    with torch.no_grad(), pytest.raises(NotImplementedError, match="the prefill arm is deleted"):
        rola.rola_op(routes, v)


def test_a_cpu_bundle_is_refused_because_there_is_no_cpu_lowering():
    """The arm is a CUDA kernel and there is no CPU implementation to reach."""
    v, routes = _tiny_routing(requires_grad=False)
    with torch.no_grad():
        assert "CUDA kernel" in envelope_refusal(routes, v)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_envelope_admits_a_no_grad_cuda_bundle_on_a_built_arm():
    """The other side of the same decision, which a CPU-only run cannot see: with the
    gradient off and the operands on the device, an in-matrix bundle must actually be
    ADMITTED. Asserting only the refusals would let the arm rot behind them."""
    torch.manual_seed(5)
    producer = build_routes(hidden_size=32, num_heads=1, widths=(8, 8),
                           routing=_dense_routing()).to("cuda")
    x = torch.randn(1, 32, 32, device="cuda")
    v = torch.randn(1, 32, 1, 64, device="cuda")
    with torch.no_grad():
        routes = producer(x)
        assert envelope_refusal(routes, v) is None
        require_envelope(routes, v)


# --- the O-3 name contract ---------------------------------------------------

_PUBLIC_V3_EXEMPTIONS = {
    # None. If this set ever gains an entry, the entry must say why the name is
    # public AND must say `v3`, which is a hard thing to justify.
}


def test_no_public_name_in_the_package_says_v3():
    """O-3, checked mechanically over the package's own public surface: every
    module-level `def`/`class`/assignment whose name does not start with `_`, plus
    every `__all__` entry. Comments and docstrings are out of scope -- they cite the
    spec document `ROLA_V3_SPEC.md` and the `rola-v3` source branch, which are
    real proper nouns and not spellings of any function."""
    offenders = []
    for path in sorted(pathlib.Path(rola.__path__[0]).rglob("*.py")):
        tree = ast.parse(path.read_text())
        names = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Assign):
                names.extend(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.append(node.target.id)
        for name in names:
            if name.startswith("_"):
                continue
            if "v3" in name.lower() and name not in _PUBLIC_V3_EXEMPTIONS:
                offenders.append(f"{path.relative_to(rola.__path__[0])}::{name}")
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
                    and isinstance(node.value, (ast.List, ast.Tuple))):
                for element in node.value.elts:
                    if (isinstance(element, ast.Constant) and isinstance(element.value, str)
                            and "v3" in element.value.lower()):
                        offenders.append(f"{path.relative_to(rola.__path__[0])}::__all__:{element.value}")
    assert not offenders, f"public names still saying v3: {offenders}"


def test_the_top_level_exports_are_exactly_the_declared_api():
    """`__all__` is the contract; a name that is exported but not declared is a name
    nobody promised and everybody will import anyway."""
    for name in rola.__all__:
        assert hasattr(rola, name), f"{name} is in __all__ but not importable"
    assert "rola_op" in rola.__all__
    assert not any(name.startswith("_") and name != "__version__" for name in rola.__all__), (
        "a private name must not be re-exported: the op's public spelling is rola_op")
