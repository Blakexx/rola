#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CAMPAIGN WORK-CODE CHECK, run on its own; `tools/lint/ratchet.py work_codes` gates it.

The rule is `lint_standards.check_no_campaign_work_codes` and it is defined there, once: this file is a front door, not
a second implementation. A commit may not add a work code. The ones the tree still carries are the ratchet's baseline,
most of them in the deletions ledger and the open-work list, whose rows need to say what each change DID rather than
which stage did it -- prose work, not a substitution -- and the rest in test and tool docstrings.

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
    print(f"work_codes: {len(findings)} finding(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
