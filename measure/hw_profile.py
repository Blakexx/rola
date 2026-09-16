"""A HARDWARE PROFILE, never a hostname: the GPU model and its SM architecture.

`NVIDIA GeForce RTX 3080 Ti` + `sm_86` becomes `rtx3080ti-sm86`; `NVIDIA A100-SXM4-80GB` + `sm_80` becomes
`a100-sm80`, because the memory-config suffix after the first `-` is a fact about the SKU, not about which run this
is.
"""
from __future__ import annotations

import re

#: Vendor words carry no comparability information; stripping them is what turns
#: `torch.cuda.get_device_properties(0).name` into a slug instead of a sentence.
_VENDOR_WORDS = ("NVIDIA", "GeForce", "Tesla", "Quadro")


def slugify_gpu(name: str) -> str:
    """`NVIDIA GeForce RTX 3080 Ti` -> `rtx3080ti`; `NVIDIA A100-SXM4-80GB` -> `a100`.

    The cut at the first `-` drops a memory/interconnect suffix (`SXM4-80GB`,
    `PCIE-40GB`) that does not change which kernel binary applies -- an A100-40GB and
    an A100-80GB run the identical SASS.
    """
    stripped = name
    for word in _VENDOR_WORDS:
        stripped = stripped.replace(word, "")
    stripped = stripped.split("-", 1)[0]
    return re.sub(r"[^A-Za-z0-9]", "", stripped).lower()


def current() -> str:
    """This machine's profile, read from the driver (`nvidia-smi`), for a writer that does not hold a CUDA context."""
    import subprocess

    line = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.splitlines()[0]
    name, cap = (x.strip() for x in line.split(","))
    return hw_profile(name, "sm_" + cap.replace(".", ""))


def hw_profile(gpu_name: str, sm: str) -> str:
    """The results-tree key: `<gpu-slug>-<sm>`, e.g. `rtx3080ti-sm86`."""
    return f"{slugify_gpu(gpu_name)}-{sm.replace('_', '')}"
