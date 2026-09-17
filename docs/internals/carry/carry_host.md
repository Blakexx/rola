# `carry/carry_host.cuh` — the call's derivation

Mirrors `csrc/rola/src/carry/carry_host.cuh`.

One derivation of a carry call, shared by the launch (`carry.cu`) and by the part harness
(`measure/harness/carry_parts/carry_parts.cu`), so a part runs on exactly the call the
kernel would: the kernel boundary's refusals (`refuse_shape`, `refuse_schedule`,
`refuse_operand`, the carve order), the addressing block, both sides' layouts
(`make_side_layout`), and the parameter block filled but for the phase ledger and trace
(`derive_carry_call` → `CarryCall`). A second derivation anywhere would be a second
definition of the call, which is how a harness comes to disagree with the kernel.
