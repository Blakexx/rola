"""THE CARD'S PRODUCT (docs/internals/common/select.md#slug): the binary finds a device's product by the slug of its
name (`select.cu`'s `slug_of`), and the generators and the measurements by `measure/hw_profile.py`'s `slugify_gpu`.
The two are one rule written twice, on either side of the extension; this ties them on the machine's own device."""
from __future__ import annotations

import torch

from measure import hardware
from measure.hw_profile import slugify_gpu
from rola.ops.window import card_of


def test_the_binary_finds_the_product_the_host_names():
    """The card's product row is the one `slugify_gpu` names among the generated `kProducts` (sorted by slug), or
    none where the device has no row; its variant is that row's, else the architecture's own."""
    products = sorted(hardware.load()["product"])
    slug = slugify_gpu(torch.cuda.get_device_name())
    card = card_of()
    assert card.product == (products.index(slug) if slug in products else -1), (slug, card)
    assert card.variant == (hardware.variant_of(slug) if slug in products else 0), (slug, card)
