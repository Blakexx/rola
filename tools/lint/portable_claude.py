#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE AGENT FILES ARE PORTABLE (KERNEL_STANDARDS section 31, gating): nothing under `.claude/` and no `CLAUDE.md`
names a machine's own path -- a home directory, a mount, a drive. Hooks resolve through `$CLAUDE_PROJECT_DIR`, skills
cite repo-relative paths, and a site's own locations come from the dev config or the environment (`ROLA_SCRATCH`), so a
clone or a worktree anywhere runs the same agent setup. Docs: docs/internals/tools/lint.md#portable-agent-files.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: a machine's own location: a home directory, a WSL or macOS mount, a Windows drive
MACHINE = re.compile(r"(?<![\w$])(/home/[^/\s]+|/Users/[^/\s]+|/mnt/[a-z]/|/root/|[A-Za-z]:\\)")


def agent_files() -> list[str]:
    """The tracked files agents load: everything under `.claude/` and every `CLAUDE.md`."""
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f.startswith(".claude/") or f.split("/")[-1] == "CLAUDE.md"]


def main() -> int:
    found = []
    for name in agent_files():
        try:
            text = (ROOT / name).read_text()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if m := MACHINE.search(line):
                found.append(f"{name}:{number}: `{m.group(1)}` is a machine's path")
    for line in found:
        print(line)
    if found:
        print(f"portable_claude: {len(found)} machine path(s) in the agent files (KERNEL_STANDARDS 31: "
              "`$CLAUDE_PROJECT_DIR`, repo-relative paths, or the dev config)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
