# `tools/life_ranges.py` — peak live registers per source region

`nvdisasm --print-life-ranges` prints, beside every SASS instruction, the number of registers
live there. This tool sums that into a report: the kernel's peak live count and its histogram,
the peak and mean per source region (the innermost function the instruction inlines from, struct
methods as `Struct::method`, the naming `tools/region_ledger.py` uses), and the peak program
points. The cubin needs `-lineinfo` for the regions; an extension's cubin (`--so`, `--member`)
reports the peak and histogram without them.

Why it exists (KERNEL_STANDARDS §22): a budget argument about registers cites the peak live at
the program point it concerns, never ptxas's allocation. The P81 fold's two-deep form was
abandoned at "255 registers allocated" while its HMMAs ran at 127 live of 255; the allocation
was set by the fill's and the head's unrolled address arrays (162 and 160 live), which bind no
MMA phase. The report makes that visible in one run.

`--arm N` compiles the carry arm's translation unit alone with `-lineinfo` against the build's generated headers
(`build/generated`, so the parts mask of the last build) for this machine's GPU architecture (or `--arch`), then reads
it -- the compile the build ledger and the composer each used to spell. `--json PATH` also writes the instruction count,
the peak live registers, whether line info was present, and every region's peak, mean and instruction count.
