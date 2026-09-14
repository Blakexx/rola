#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CAMPAIGN WORK-CODE CHECK, run on its own. REPORT-ONLY, and here is why.

The rule is `lint_standards.check_no_campaign_work_codes` and it is defined there, once:
this file is a front door, not a second implementation. It exists because the rule is
NOT YET CLEAN on this tree, and a gating hook that a contributor cannot satisfy is a
hook they learn to bypass.

WHAT IS CLEAN AND WHAT IS NOT, measured rather than promised: `csrc/` and `rola/` --
every kernel source and the whole shipped package -- carry no work code, and neither do
the tools, the tests or the benches. What remains is documentation, and most of it is in
two files whose subject IS the campaign's own history: the deletions ledger and the
open-work list. Those need a rewrite that says what each row DID rather than which stage
did it, which is prose work, not a substitution.

The flip is therefore a change of exit code and nothing else, the day the inventory this
prints reaches zero.

    python tools/lint/work_codes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lint_standards import check_no_campaign_work_codes  # noqa: E402


def main() -> int:
    findings = check_no_campaign_work_codes()
    for finding in findings:
        print(f"  {finding}")
    print(f"work_codes: {len(findings)} finding(s) -- REPORT ONLY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
