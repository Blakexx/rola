# `carry/ampere/plan.cuh` — one arm's plan

Mirrors `csrc/rola/src/carry/ampere/plan.cuh`: the box's tiles, registers, pool rows and shared memory on the compiling card.
Every compiled size is fixed by the arm and the card, never by the data (R22); what the data sizes -- each window's
chunks and buffers -- the window's plan places in the area at run time (`schedule.md`).

<a id="tenants"></a>
## The persistent tenants

`Tenants` holds what lives across the window and is the same on every card, arm and call: 8,320 bytes at W = 512,
the area from 8,320:

- both sides' counts; the write side's union tokens (1 KB) and its group's boundaries in prefix form (528 bytes), two
  windows' each by parity: a window's are copied while the window before it is still read;
- the write lists' exponent pairs, by parity;
- the unit book (a list's running unit, `fold.md#units`) and the window's scale a list (`fold.md#units`), the
  digit keys (by parity), the digit exponents in lane order a row a warp and the scales' bits (`handoff.md#maxima`);
- the den image (2 KB, `handoff.md#den`; a dense fold's exchange while it folds, `fold.md#dense`) and each digit's den
  factor and largest den (the den state is registers, `fold.md#state`);
- the phase clock's rows;
- the CTA's addresses (`Addr`, 256 bytes reserved: `parts.md#place`).

The AREA starts where they end (`kAreaAt`, the same on every call) and ends where the read tables start.

<a id="read-tables"></a>
## The read tables

What scales with the read side's query heads a write head (`g`, grouped queries) is laid out by the host for each
call (`tables_of`, into `CarryParams::tables`) at the top of shared memory, each two windows' by parity: the read union
(`g` KB), its group's boundaries in prefix form (`528 g` bytes: 33 rows of `8 g` columns), each query head's tokens'
scale bytes (`512 g` bytes, the scale pass's, `readout.md#chunk`) and the plan records (`plan_words(g)` words,
`schedule.md#plan`). One arm serves every `g`; the area is what the tables leave, 88,832 bytes at one head on sm_86 (the
census), and the launch refuses a `g` whose area does not hold the least plan (`least_plan_fits`, at the call's reduce
dtype). At the top, not after the tenants, because a lane's address in the area folds the
area's base into its instruction as an immediate: from a base read at run time each such address held a register,
and at 255 a thread they spilled.

<a id="barriers"></a>
## The barriers

None in shared memory since 2026-10-25: the async copies land on commit groups in a fixed order a window
(`carry_kernel.md#staging`), and every rendezvous is the hardware barrier (`carry_kernel.md#barriers`). The mbarriers
the step machine landed its two slots on, each counting all 256 threads, are retired with it.

<a id="box"></a>
## The box

An arm's box is its plan's, from its key: the fold's rows `kRows` (`fold_rows`) and the channels `kChannels`
(`channels`), the fold's lists `kLists` the state's rest (`kStateValues / (kRows kChannels)`). The readout's digits
`kDigits` and its K `kK` are the fold's rows and lists in the alternating form, its lists and rows in the same split
(`kSameSplit`: the fold's lists are the readout's digits); the key's `read_rows` is asserted the K. From them: the
fold's tile (`kFoldLists` = `kLists / 8` a warp, `kFoldMTiles` = `kRows / 16`, `kFoldNTiles` = `kChannels / 8`), the
readout's warps (`kChannelGroups` = `kChannels / 8`, `kTokenGroups` the warps left over, `kDenWords` one a channel
group), the image below and the pool rows. The fixed tenants are sized at the caps (`kMaxExtent`, `../carry.md`), so an
arm's box never moves them. Each component asserts the extents it serves, and the plan the boxes they serve together:
16 or 32 lists and rows, 32 or 64 channels -- the shipped 32 x 32 x 32 boxes, the alternating 32 x 16 x 64 (one token
group of eight channel groups), a dense write side's 16 x 32 x 64 (two fold lists a warp, `fold.md#dense`) and the same
split's 16 x 32 x 64 (both sides listing a level of 16, a listed fold of two lists a warp) -- the boxes the planner
picks (`../../engine/carve.md#producer-v1`); a batch is half an m-tile of every fold list (the alternating form) or a list of every
warp's (the same split).
A pool row: the read side's slice over the readout's K, the write side's over the fold's rows (a dense write side's
second over its lists), then the write side's values.

<a id="list-groups"></a>
## The list groups

A side's LIST GROUP is the digits of its list level one union of the facts pass spans, with its prefix rows
(`../../facts/facts.md#list-groups`): `kReadListGroup`, `kWriteListGroup`, what the arm reads its lists in, so the
facts pass is called in them (`rola.ops.carry.lists_form`): the readout's digits and the fold's lists, 32 at the
32 x 32 x 32 boxes and the read side's 16 at 32 x 16 x 64 -- a union of exactly the box's own lists, its prefix rows
`G + 1` inside the tenants' `kPreRows` cap. The host checks the lists' shapes and a listed level's extent against the
arm's groups (`../carry.md#refusals`).

<a id="image"></a>
## The image

`kPlaneBytes` a readout digit (its `kK` lists x `kChannels` channels, f16; 2 KB at 32 x 32), the whole image
`kImageWhole` (`kDigits` planes, 64 KB) or `kImageRing` (`kHalfPlanes` = half of them, 32 KB): one half of two, stored
and read by each read chunk (`schedule.md#buffers`), a plane its channel groups' slices (`kSliceBlock` bytes each,
`handoff.md#image`). `kBatchPlanes` = 8 digits a batch (`kBatches` = `kDigits / 8`): half an m-tile of each of a
warp's lists in the alternating form, one of its lists in the same split (`handoff.md#store`).

<a id="plan"></a>
## The plan

`Plan<Arm, B>` is an arm's key (`tools/gen_shards.py`'s `ARM_KEY`) at build `B`
([`../../common/hardware.md#builds`](../../common/hardware.md#builds)): its window, list boundaries and caps are the build's
traits, so the host and the device read one plan. The key fixes each side's dense rows a warp and its operand's level count, and the hand-off's form
(`kSameSplit`, `handoff.md#same-split`) and the box (`#box`); the plan asserts a listed read side (the dense readout is
the next component), and derives:

- the fold's tiles: a warp folds lists `w + 8 j` (4); the readout's warps are its token and channel groups
  (`readout.md#warps`);
- the registers: `kStateRegs` = 128 fp32 a lane;
- the pool rows: a read row is the read side's operand slices (`kReadRow`: 64 bytes, or 128 with a coefficient level;
  a dense read side's K rows then its digits' coefficients, 96 at `[16, 2, 16, 2, 64, 1]`: `readout_dense.md#row` --
  a dense read side only where its digits are the fold's lists, the same split), a write row the write side's slices
  then the values (`kWriteRow`: 128 to 224);
- the shared memory: `kSmemBytes`, the card's whole opt-in maximum a CTA, `kAreaAt` where the area starts, and
  `kArea`, what the tenants and one head's read tables leave (the census's) -- 88,832 bytes on sm_86.

The launch sizes for the device's card, and the kernel refuses any other size (`../carry_arm.md#plan`).
