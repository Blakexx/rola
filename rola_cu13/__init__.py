# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""rola's binary plugin for the cu13 toolchain: the extension ``_C``, the record of its build ``_build_config`` and the
manifests it was ratified against (``manifests/``, in a wheel).

``rola`` is pure Python and loads this package when a kernel is first called (``rola.ops._ext``); nothing else imports
it. It installs as ``pip install "rola[cu13]"``, at exactly ``rola``'s version. A checkout's in-place build lays out the
same package, so an installed wheel and a checkout load the binary by one path (docs/build.md#wheels).
"""
