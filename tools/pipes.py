# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PIPES A LAUNCH TAKES: one set of ncu counters for any kernel's launch and its pricing per card -- each pipe's
cycles (tensor, the shared-memory pipe by kind of traffic, issue, the integer, FMA and transcendental pipes, the SM's
global path: L2's lines and its atomic requests) at
`common/hardware.json`'s rates, the largest binding; and, on the card it ran on, each pipe's busy share of the elapsed
cycles, L2 sectors and the stall reasons. The lab (`tools/lab.py run`) reads it for every experiment kernel, the
production capture (`tools/pipe_counters.py`) for a carry launch. Stdlib only. docs/internals/tools/pipes.md
"""
from __future__ import annotations

#: THE COUNTERS, by the short name the pricing reads (ncu's metric names): instructions by pipe, the shared-memory
#: pipe's traffic by kind, L2 sectors, elapsed cycles and the measured busy shares
METRICS = {
    "hmma": "sm__inst_executed_pipe_tensor_op_hmma.sum",
    "imma": "sm__inst_executed_pipe_tensor_op_imma.sum",
    "tensor_counted": "smsp__pipe_tensor_op_hmma_cycles_active.sum",
    "tensor_counted_imma": "smsp__pipe_tensor_op_imma_cycles_active.sum",
    "inst": "smsp__inst_executed.sum",
    "alu": "sm__inst_executed_pipe_alu.sum",
    "fma": "sm__inst_executed_pipe_fma.sum",
    "fma_fp16": "sm__inst_executed_pipe_fma_type_fp16.sum",
    "fmaheavy": "sm__inst_executed_pipe_fmaheavy.sum",
    "xu": "sm__inst_executed_pipe_xu.sum",
    "wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "load_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "stores": "sm__sass_inst_executed_op_shared_st.sum",
    "store_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "atomics": "sm__sass_inst_executed_op_shared_atom.sum",
    "atomic_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum",
    "copy_bypass_bytes": "sm__sass_data_bytes_mem_shared_op_ldgsts_cache_bypass.sum",
    "copy_access_bytes": "sm__sass_data_bytes_mem_shared_op_ldgsts_cache_access.sum",
    "global_reads": "l1tex__data_pipe_lsu_wavefronts_mem_lg_cmd_read.sum",
    "global_fill_bytes": "l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum",
    "l2_read": "lts__t_sectors_srcunit_tex_op_read.sum",
    "red_requests": "lts__t_requests_srcunit_tex_op_red.sum",
    "l2_write": "lts__t_sectors_srcunit_tex_op_write.sum",
    "elapsed": "sm__cycles_elapsed.avg",
    "elapsed_sum": "sm__cycles_elapsed.sum",
    "busy_issue": "smsp__issue_active.avg.pct_of_peak_sustained_elapsed",
    "busy_alu": "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "busy_fma": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "busy_fmaheavy": "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_elapsed",
}
STALLS = ("long_scoreboard", "short_scoreboard", "mio_throttle", "math_pipe_throttle", "barrier", "wait",
          "not_selected", "lg_throttle", "dispatch_stall", "branch_resolving", "no_instruction", "membar", "drain",
          "sleeping", "tex_throttle", "imc_miss", "misc")

#: ncu's tensor-pipe active cycles a scheduler for one m16n8k16 (an m16n8k8 half), by the architecture it ran on,
#: whatever the accumulate: a count of the work, not its time (sm_86's fp32 accumulate takes 32; an
#: `HMMA.16816.F32.BF16` counts 16.0) -- so tensor time is priced from it, never read as the profiler's ratio (§22)
TENSOR_COUNTED = {"sm_86": 16.0}

#: an MMA mnemonic's prefix and suffix and the `hardware.json` kind it runs at, the first match winning
MMA_KINDS = (("HMMA", ".F32.BF16", "hmma_bf16_f32"), ("HMMA", ".F32", "hmma_f16_f32"), ("HMMA", ".F16", "hmma_f16_f16"),
             ("IMMA", "", "imma_u8_s8_k32"))


def stall_metric(reason: str) -> str:
    return f"smsp__warp_issue_stalled_{reason}_per_warp_active.pct"


def metric_names() -> list[str]:
    """Every ncu metric the set reads."""
    return [*METRICS.values(), *map(stall_metric, STALLS)]


def short(values: dict[str, float]) -> dict:
    """A launch's counters by short name (`METRICS`), absent ones 0, with `stalls` by reason, from ncu's values by
    metric name."""
    c: dict = {name: values.get(metric, 0.0) for name, metric in METRICS.items()}
    c["stalls"] = {r: values.get(stall_metric(r), 0.0) for r in STALLS}
    return c


def mma_kind(mnemonics: dict[str, int]) -> str | None:
    """The one MMA kind a kernel's SASS issues (`MMA_KINDS`), None for none or several."""
    kinds = {next(kind for op, suffix, kind in MMA_KINDS if m.startswith(op) and suffix in m)
             for m in mnemonics if m.startswith(("HMMA", "IMMA"))}
    return kinds.pop() if len(kinds) == 1 else None


def sms(c: dict) -> float:
    return c["elapsed_sum"] / c["elapsed"] if c["elapsed"] else 1.0


def shared_cycles(c: dict, rates: dict) -> float:
    """The shared-memory pipe's cycles a launch takes (summed over its SMs), its traffic by kind at the card's prices
    (docs/internals/common/hardware.md#shared): load wavefronts; stores and atomics each at the larger of their
    instructions' least and their wavefronts; copies into shared memory and global loads by their bytes."""
    r = {k: v.cycles for k, v in rates.items()}
    return (c["load_wavefronts"] * r["wavefront"]
            + max(c["stores"] * r["shared.store"], c["store_wavefronts"] * r["shared.store_wavefront"])
            + max(c["atomics"] * r["atomic"], c["atomic_wavefronts"] * r["atomic_wavefront"])
            + c["copy_bypass_bytes"] / 128 * r["shared.copy_bypass"]
            + c["copy_access_bytes"] / 128 * r["shared.copy_access"]
            + c["global_reads"] * r["shared.global_read"]
            + c["global_fill_bytes"] / 128 * r["shared.global_fill"])


def pipe_cycles(c: dict, n: float, rates: dict, arch_ran: str, mma: tuple[str, int] | None,
                kind: str | None) -> dict[str, float]:
    """KERNEL_STANDARDS §29.14, one card: the cycles an SM's pipes spend a unit `n` of the launch (its counts summed
    over its SMs) -- the tensor pipe (a declared MMA's kind and K, else the m16n8k16s counted on `arch_ran`, at the
    card's rate for the SASS's one kind), issue, the integer, FMA and transcendental pipes (each instruction at its
    pipe's cycles a scheduler), all over the schedulers, and the shared-memory pipe (one an SM); the largest binds."""
    r = {k: v.cycles for k, v in rates.items()}
    s = r["schedulers"]
    out = {}
    if mma:
        out["tensor"] = (c["hmma"] + c["imma"]) / n * r[f"mma.{mma[0]}"] * mma[1] / 16 / s
    elif kind and c["hmma"] + c["imma"] and arch_ran in TENSOR_COUNTED:
        k16 = (c["tensor_counted"] + c["tensor_counted_imma"]) / TENSOR_COUNTED[arch_ran]
        out["tensor"] = k16 / n * r[f"mma.{kind}"] / s
    out["shared"] = shared_cycles(c, rates) / n
    out["issue"] = c["inst"] / n * r["issue"] / s
    out["alu"] = c["alu"] / n * r["pipe.alu"] / s
    out["fma"] = max((c["fma"] - c["fma_fp16"]) * r["pipe.fma"] + c["fma_fp16"] * r["pipe.fma_fp16"],
                     c["fmaheavy"] * r["pipe.fmaheavy"]) / n / s
    out["xu"] = c["xu"] / n * r["pipe.xu"] / s
    out["global"] = global_cycles(c, rates) / n
    return out


def global_cycles(c: dict, rates: dict) -> float:
    """The SM's global path a launch takes (summed over its SMs), at the card's prices with every SM busy
    (docs/internals/common/hardware.md#global): each 128-byte line read from or written to L2, and each global atomic
    request."""
    r = {k: v.cycles for k, v in rates.items()}
    return (c["l2_read"] + c["l2_write"]) / 4 * r["global.l2_line"] + c["red_requests"] * r["global.red_request"]


def busy_here(c: dict, rates: dict, arch_ran: str, mma: tuple[str, int] | None, kind: str | None) -> dict[str, float]:
    """On the card the launch ran on (`rates` its own): each pipe's busy percent of the elapsed cycles -- tensor and
    the shared-memory pipe priced (`pipe_cycles`), issue and the integer and FMA pipes measured."""
    priced = pipe_cycles(c, sms(c), rates, arch_ran, mma, kind)
    whole = max(c["elapsed"], 1.0) / 100
    return {"issue": c["busy_issue"], "tensor": priced.get("tensor", 0.0) / whole, "alu": c["busy_alu"],
            "fma": c["busy_fma"], "fmaheavy": c["busy_fmaheavy"], "shared": priced["shared"] / whole,
            "global": priced["global"] / whole}
