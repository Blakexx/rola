"""The warp timeline of one cell: one CTA's warps, window by window, each activity with its real cycles (the kernel's
stamps, an uninstrumented launch) and the instructions it executed (the warp trace, an instrumented launch of the same
binary), matched by (window, event, ordinal). KERNEL_STANDARDS §22 (12).

    python tools/warp_timeline.py CELL --json timeline.json [--html timeline.html] [--cta 0] [--keep-trace]

The JSON is the instrument's record: per warp, per window, every interval's event, cycles and instruction mix, plus
the window summaries (each scheduler's pipe-fed share, the cycles and instructions by activity). The page draws it
(`tools/timeline_page.py`); several cells' JSONs draw on one page through `--html` with `--also`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dev_config  # noqa: E402
import warp_trace  # noqa: E402
from phase_trace import HMMA_CYCLES, SCHEDULERS, take_stamps  # noqa: E402

BINS = 400


def summaries(warps: list[dict]) -> list[dict]:
    """Per window: each scheduler's pipe-fed share (an interval's HMMAs times the pipe's cycles spread over it, the
    scheduler's two warps summed and capped at one) and the cycles, events and instructions by activity."""
    out = []
    for wi in range(min(len(w["windows"]) for w in warps)):
        length = max(w["windows"][wi]["len"] for w in warps)
        fed = []
        for s in range(SCHEDULERS):
            occ = [0.0] * BINS
            for w in (s, s + SCHEDULERS):
                for iv in warps[w]["windows"][wi]["ivs"]:
                    h = (iv["mix"] or {}).get("hmma", 0)
                    if not h:
                        continue
                    d = min(1.0, h * HMMA_CYCLES / max(1, iv["t1"] - iv["t0"]))
                    for b in range(int(iv["t0"] / length * BINS), min(BINS, int(iv["t1"] / length * BINS) + 1)):
                        occ[b] = min(1.0, occ[b] + d)
            fed.append(sum(occ) / BINS)
        acc: dict[str, dict] = defaultdict(lambda: {"cycles": 0, "events": 0, "instructions": 0, "hmma": 0})
        for w in warps:
            for iv in w["windows"][wi]["ivs"]:
                a = acc[iv["ev"]]
                a["cycles"] += iv["t1"] - iv["t0"]
                a["events"] += 1
                a["instructions"] += (iv["mix"] or {}).get("n", 0)
                a["hmma"] += (iv["mix"] or {}).get("hmma", 0)
        n = len(warps)
        out.append({"cycles": length, "pipe_fed": fed,
                    "by_activity": {k: {"cycles_a_warp": v["cycles"] / n, "events_a_warp": v["events"] / n,
                                        "instructions_an_event": v["instructions"] / max(1, v["events"]),
                                        "hmma_an_event": v["hmma"] / max(1, v["events"])} for k, v in acc.items()}})
    return out


def take(cell: str, cta: int, keep: bool, addrs: bool, census: Path | None = None) -> dict:
    import sass

    arch = sass.device_arch()
    work = dev_config.scratch("warp_timeline")
    stem = work / f"{cell}-cta{cta}"
    records, listing, child_stamps = stem.with_suffix(".bin"), stem.with_suffix(".tsv"), stem.with_suffix(".child.pt")
    real, warps, _ctas = take_stamps(cell, cta + 1, 32768)
    warp_trace.record(cell, cta, records, listing, child_stamps, arch, addrs=addrs)
    import torch

    recs, sasslist = warp_trace.load(records, listing)
    matched, by_event, total = warp_trace.match(recs, sasslist, torch.load(child_stamps)[cta], real[cta], warps)
    if census is not None and str(census) == "auto":
        census = dev_config.scratch("stall_census") / f"{cell}.csv"
        if not census.exists():
            raise SystemExit(f"no census export for {cell} at {census}: run tools/stall_census.py {cell} first")
    joined = warp_trace.census_join(census, sasslist, by_event, total, arch) if census else None
    if keep:
        torch.save(real, stem.with_suffix(".real.pt"))
    else:
        for f in (records, listing, child_stamps):
            f.unlink(missing_ok=True)
    unmatched = sum(1 for w in matched for win in w["windows"] for iv in win["ivs"] if iv["mix"] is None)
    return {"cell": cell, "cta": cta, "arch": arch, "warps": matched, "windows": summaries(matched),
            "unmatched_intervals": unmatched, "nvbit": warp_trace.pin()["version"], "census": joined}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--cta", type=int, default=0, help="the CTA traced (its stamps are taken on the same launch index)")
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--html", type=Path, default=None, help="also draw the page")
    ap.add_argument("--also", type=Path, action="append", default=[], help="other timeline JSONs drawn on the page")
    ap.add_argument("--keep-trace", action="store_true", help="keep the raw records (tens of MB) in the scratch")
    ap.add_argument("--addrs", action="store_true", help="record every memory operand's lane addresses")
    ap.add_argument("--census", type=Path, default=None,
                    help="a `stall_census.py` export of this cell, its samples joined into the activities; `auto` takes "
                         "the export the census keeps in its scratch")
    a = ap.parse_args()
    tl = take(a.cell, a.cta, a.keep_trace, a.addrs, census=a.census)
    a.json.write_text(json.dumps(tl))
    w = tl["windows"]
    fed = statistics.mean(statistics.mean(x["pipe_fed"]) for x in w)
    print(f"{a.cell}: CTA {a.cta}, {len(w)} windows, {statistics.median(x['cycles'] for x in w):.0f} cycles a window "
          f"(median), pipe fed {100 * fed:.0f}% (HMMA estimate), {tl['unmatched_intervals']} intervals without a mix")
    if tl["census"]:
        c = tl["census"]
        print(f"census joined: {len(c['instructions'])} traced instructions, {len(c['unjoined'])} without a profiler row")
    if a.html:
        import timeline_page

        pages = {a.cell: tl}
        for other in a.also:
            o = json.loads(other.read_text())
            pages[o["cell"]] = o
        a.html.write_text(timeline_page.render(pages))
        print(f"page: {a.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
