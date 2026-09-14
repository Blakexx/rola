# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""Operators: the consumers, the fp64 oracle, decode, and paging.

`naive` ships in the wheel deliberately — it is the definition the kernel is
gated against, so a user can check us on their own shapes and hardware without a
repository checkout. See docs/testing.md.
"""
