#!/usr/bin/env python3
"""VULTURE WHITELIST (LINT2 item 4, G_FOUNDATION G5, REPORT-ONLY until G5).

vulture's own convention: a whitelist is Python source that REFERENCES every
name vulture would otherwise flag, so it disappears from the report; it is
never executed, only parsed. Every entry below is a GENUINE DYNAMIC USE --
a name some FRAMEWORK reads by convention, never by an explicit call vulture's
static analysis can see -- with a one-line reason, per the brief. Anything
NOT listed here that vulture still flags is a real candidate for G5 to delete,
not something this stage silences by omission.

Run: `vulture rola/ tools/ tests/ benchmarks/ tools/lint/vulture_whitelist.py`
(`tools/lint/run_vulture.sh` does exactly this, report-only).
"""


class _Whitelist:
    """vulture's own idiom (see its README "Handling false positives"): a
    referenced ATTRIBUTE never triggers "unused", so every genuinely-dynamic
    name is listed here as `_.name` -- a bare module-level assignment
    (`pytestmark = None`) does NOT work, it is itself a new "unused variable"
    (measured: tried first, vulture flagged this very file)."""

    # pytest reads a module-level `pytestmark` attribute automatically to apply
    # markers (e.g. `pytest.mark.skipif(...)`) to every test in the file; it is
    # never referenced by name anywhere in the file itself, which is exactly
    # what vulture's "unused variable" heuristic flags. Present in every
    # tests/{oracle,integration,unit}/test_*.py file that needs a module-wide
    # skip or GPU marker (27 files, measured 2026-08-29).
    pytestmark = None

    # pytest FIXTURE INJECTION: a test function's parameter name is matched by
    # NAME against a `@pytest.fixture` of the same name and the framework calls
    # it for you -- the parameter is "unused" only to a static analyzer that
    # does not know about fixture injection. `tests/unit/test_ratify_argv.py`'s
    # `ninja` fixture (patches `ratify._ninja_cuda_template` for every test in
    # the file) is the one vulture flags at 60%+100% confidence (the fixture
    # function itself, and each test parameter named after it).
    ninja = None


_ = _Whitelist()
_referenced = (_.pytestmark, _.ninja)  # ruff B018: a bare attribute expression is "useless"
