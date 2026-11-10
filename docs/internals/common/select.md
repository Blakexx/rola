# `common/select.cuh` — the selection layer

Mirrors `csrc/rola/src/common/select.{cu,cuh}`.

Every kernel adapts to its card (the constitution's R8; [`KERNEL_STANDARDS.md` section
8](../../KERNEL_STANDARDS.md#8-arch-mapping-at-the-atom-closed-world-at-the-arm)): its sizes come from its plan at a
BUILD ([`hardware.md#builds`](hardware.md#builds)), and the host reaches the build of the device it holds through
this one layer -- the device's card, the builds the binary carries, one branch to the card's build at compile time --
rather than a switch over compute capabilities in each kernel.

<a id="card-of"></a>
## `card_of(device)`, `current_card()`

A device's `Card`: its compute capability, its product row (`kProducts`' index, -1 where the device has none), its SM
count (the driver's: a product's SKUs differ in it, so no table holds it), the build it runs and that build's traits. The first call for a device ADMITS it -- the closed-world checks
([`arch_runtime.md#check-arch-table`](arch_runtime.md#check-arch-table)): its architecture tabulated, the
driver's numbers its row's, and its build one this binary carries -- and keeps the card; every later call is a lookup.
A device whose build the binary does not carry is refused by name with the builds it does carry, so an architecture is
never run on another's image (an sm_89 card is not run on the sm_86 image: it was not measured there).

<a id="slug"></a>
The product is found by the slug of the device's name, `measure/hardware.py`'s `slugify_gpu` (the vendor words
dropped, cut at the first `-`, alphanumerics lowercased): every A100 memory and interconnect variant is `a100`. The
product's row names its build variant; a device without a row runs its architecture's variant 0.

<a id="carried"></a>
## The builds a binary carries

`build/generated/builds.inc` (`ROLA_BUILDS(F)`, one `F(cc, variant)` a build) is written before every compile by
the graph's header action and by `tools/ratify.py`'s compile path from the build's architectures (`measure/hardware.py`'s `builds_header`): every
variant of every architecture the binary compiles.

<a id="in-image"></a>
## `in_image(b)`

A build's kernels compile in its own image only. `in_image(b)` is true for every build in the host pass and for the
image's own build in a device pass (`b.cc == __CUDA_ARCH__`), and a launch site instantiates a build's kernels under
`if constexpr (in_image(B))` -- so the sm_80 image holds no sm_86 kernel and the binary's entry points are one a
kernel a build, with no trapping copies in the other images. A launch of another image's kernel is refused by the
runtime (an invalid device function) and, before that, by `card_of`.

<a id="build-switch"></a>
## `build_switch(card, body)`

The host's one branch to the card's build: `body(BuildTag<B>{})`, `B` the card's build as a compile-time value and
`BuildTag<B>::traits` its traits, over `ROLA_BUILDS`. The carry's launch and census take their build here
([`../carry/carry_arm.md#plan`](../carry/carry_arm.md#plan)).
