# The documents

One owner per topic: a fact is stated on the page that owns it and linked from everywhere else. The docs are current
truth; what was removed, and why it was safe, is [`internals/DELETIONS.md`](internals/DELETIONS.md) and git.

| page | owns |
|---|---|
| [`ESSENTIALS.md`](ESSENTIALS.md) | the post-compaction pointer card: the laws to read before acting, the skills, the instruments |
| [`CONSTITUTION.md`](CONSTITUTION.md) | the owner's standing rulings, above every law |
| [`KERNEL_STANDARDS.md`](KERNEL_STANDARDS.md) | the laws: rule, why, enforcement, evidence |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | what the family computes, the leaf lattice, the state format, the carve, each kernel, the launch policy, portability, and the index from mechanism to page |
| [`architecture/`](architecture/README.md) | the permanent decisions and their arguments: the self-masking theorem, the affine recurrence, the factored representation, fp32 accumulation, closed-world codegen, the guarantee/choice/preference taxonomy |
| [`api.md`](api.md) | the public surface and its envelope: `rola_op`, the bundle, the routing and producer, decay, the layer, the expert pins |
| [`build.md`](build.md) | building: what the build checks, iteration subsets, the carry arm list, a kernel's identity, fatbin compression, the stable ABI, the wheels |
| [`build-graph.md`](build-graph.md) | the build graph: Bazel through its front door (the machine's caches, the host budget), the environment's layers, the graph's rules |
| [`ratification.md`](ratification.md) | the ratification manifest, the shipped-set declaration, what the gate refuses, the closed-world rules |
| [`testing.md`](testing.md) | the test tiers, the oracle and its dual run, per-slot tolerance, the gates, running the suite, the GPU lock, the sanitizer lane |
| [`measurement.md`](measurement.md) | the measurement method, its preconditions, the levels, the stopwatch, the verdict, the record, the declared build |
| [`setup.md`](setup.md) | setting up a machine and a worktree |
| [`bringup.md`](bringup.md) | bringing up a new architecture or toolkit |
| [`provenance.md`](provenance.md) | where the code came from and the related repositories |
| [`open-work.md`](open-work.md) | what is knowingly unfinished |
| [`internals/`](internals/README.md) | the per-source mirror: one page a source file under `csrc/rola/src/` and the Python modules named there, each explaining why and how pieces connect; the tools' pages under `internals/tools/` |
| [`lab/`](lab/README.md) | one page a lab experiment: the machine's measured facts the laws cite |
