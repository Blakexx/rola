# `common/arm_switch.cuh` — the host dispatch over the shipped arm set

Mirrors `csrc/rola/src/common/arm_switch.cuh`.

`uniform_switch` (`structure_switch.md`) is the DEVICE branch over a generated set;
`arm_switch` is its HOST twin. The choice it makes is which compiled body a call
reaches, so it happens once per launch on the host rather than per CTA on the device,
and the set it dispatches over is the family's arm table rather than a geometry field's
admissible shapes. The contract is otherwise the same one, and the reason is the same:
a dispatch that enumerates its own cases is a second enumeration of a set that already
exists, and two enumerations of one set drift.

<a id="carry-arm-set"></a>
## The four layers, and who owns each

1. **The cases are GENERATED, with the count asserted.** `ROLA_CARRY_ARM_SET_X(F)` in
   the generated selection header expands to one `F(index, D, DV, warps_per_cta)` per
   BUILT arm, and `CarryArmSet` is built from it — no hand-written list exists to
   disagree with the per-arm translation units, because both come out of
   `tools/gen_shards.py` reading `tools/manifests/shipped_set.json`. The header also
   carries `ROLA_CARRY_ARM_COUNT` and `ROLA_CARRY_ARM_DECLARED`.
2. **A compile-time exhaustiveness assert.** `arm_switch` static-asserts that the set
   is a subset of the declaration (`count <= declared`). A build carries FEWER arms
   than the declaration declares — never other ones — so a set larger than the
   declaration means the selection and the declaration disagree, and the translation
   units were compiled against something the declaration does not describe.
<a id="offending-field"></a>
3. **A launch-time refusal at the seam, naming the field.** The refusal says which of
   `D`, `DV`, `warps_per_cta` has no member at all, and prints both counts and the
   built set. That distinction is the actionable half: a missing `DV` is "build the
   other arm", a missing combination is "this pairing is not declared". A key matched
   by more than one member is refused separately — an arm set is a set.
4. **A trapping default and a post-build coverage sweep.** The host's trapping default
   is the refusal (there is no fallthrough), and the coverage sweep is
   `setup.py::_post_build_arm_table_check`: the freshly built `_C`, loaded by path,
   must report exactly the rows the build selected
   ([`docs/ratification.md#arm-switch`](../../ratification.md#arm-switch)).

<a id="arm-switch"></a>
## `arm_switch<ArmSet>`

`arm_switch<ArmSet>(D, dv, warps_per_cta, body)` instantiates
`body(Arm<index, D, DV, warps_per_cta>{})` once per member and runs the one whose
three fields match, so every field the body reads off its `Arm` is a compile-time
constant. Selecting an arm anywhere else — a hand-written `if` chain on `dv`, a table
of function pointers, a `switch` on a depth — is the second enumeration this exists to
prevent, and `tools/lint/drift_guards.py`'s `host_dispatch_is_arm_switch` reports one.

The set is allowed to be EMPTY, and that is not a degenerate case to be asserted away:
a binary that carries no arm of a family refuses every call to it, by name, with both
counts. That is the shipped state of the carry family, and it is what
`tests/unit/test_arm_switch.py`'s last case pins.

`ArmSet` is a type rather than a bare count so that the family's name, its key's
spelling and both counts travel with the set into the refusal. A second family adds a
second `ArmSet`, not a second switch.

## Users

- none in the shipped entry points yet: the carry family's launch surface has no
  implementation to dispatch to, and the set is empty. The primitive lands with the
  declaration, the generated set and the post-build sweep, so the body that arrives
  finds the dispatch already contracted.
- `tests/unit/test_arm_switch.py` compiles a probe against the shipped header with a
  synthetic three-arm selection header — rendered by the same renderer a build uses —
  and exercises every layer: both counts, a built arm dispatching with its own
  constants, a declared-but-unbuilt arm refused with both counts, one refusal per
  field, and the empty set refusing everything.
