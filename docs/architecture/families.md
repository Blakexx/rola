# Families: one design, a schedule per family

Every kernel runs on every product it is built for, specialized to it at compile time, and adapts to the product's
numbers rather than holding them (the constitution's R8; [`../KERNEL_STANDARDS.md` section
8](../KERNEL_STANDARDS.md#8-arch-mapping-at-the-atom-closed-world-at-the-arm)). This page is how that is organized.

## Products, builds and families

A PRODUCT is a card (its row in [`hardware.json`](../internals/common/hardware.md#families), found by the slug of its
device name). It runs a BUILD -- `Build{cc, variant}`, its architecture and, where a product's build traits differ from
its architecture's, a variant -- and a build belongs to a FAMILY: `ampere` (sm_80 to sm_89), `hopper` (sm_90),
`blackwell` (sm_100). The family is declared on the architecture's row, never derived from capability flags; products
equal in build traits share their kernels.

| what differs | where it lives | selected by |
|---|---|---|
| the structure: which warps do what, the pipeline's stages and barriers, the tiling | a SCHEDULE, `<kernel>/<family>/` | the build's family (the macro switch) |
| an instruction | an ATOM, `common/atoms/<concern>.cuh`, its body per family where it is defined | the compiling architecture's instruction flags (the micro switch) |
| a size: tiles, stages, warps, the window | the kernel's PLAN at the build (`Plan<Arm, B>`, `Walk<B>`) | the build's numeric facts and the declaration's arm |
| what the computation is | the kernel's shared semantics, `<kernel>/`, one fp64 oracle | nothing: every family's schedule is gated against it |

The CAPABILITY ROWS hold only what is read: the numeric facts the plans and the admission read and the instruction
flags the atoms read ([`../internals/common/arch_caps.md`](../internals/common/arch_caps.md)). A fact nothing reads is not
tabulated, and no flag selects a structure -- the family does.

## The entry and the card

Each kernel has ONE entry. It finds the device's card from its tensors' device (`card_of`, admitted once a process:
the architecture tabulated, the driver's numbers its row's, its build carried by this binary), branches once to the
card's build at compile time (`build_switch`), and launches the family's schedule at it; a build's kernels compile in
its own image alone (`in_image`). The caller sees one interface and passes no facts
([`../internals/common/select.md`](../internals/common/select.md)).

HARDWARE FACTS LIVE IN THE KERNEL PATH. The C++ entry sizes its outputs, its grid and its admission from the card.
Python reads a hardware fact only for a purpose the C++ path cannot serve -- the fp64 oracle building its reference
windows, the measurement tools -- and a Python read that shapes or launches a kernel call is a defect to move into the
entry.

KERNELS THAT MUST AGREE ON A FACT SHARE ITS CODE. The facts pass, the carry, the scale pass and the intra all cut the
call into the same windows: each reads W as `traits_of(B).window` at the card's build, one constant in the generated
hardware header, so they agree by construction. A value one side computes and hands to the other through Python is a
second channel that can only disagree, and is not built.
