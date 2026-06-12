"""DEPRECATED shim — the model library is now `rola.py` (we are RoLA, not CLA).

Kept only so legacy configs that still `from cla_bench import …` (incl. copies
pulled from GCS at job runtime) keep working during the transition. New code MUST
import from `rola`. Remove this file once every config has been migrated.
"""
from rola import *  # noqa: F401,F403
from rola import RoLA as ChunkedLinearAttention  # noqa: F401  (explicit legacy alias)
