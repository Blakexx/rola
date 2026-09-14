# The public mirror

This repository lives in two places: a private development repository, where every branch is pushed, and a public
repository holding what has been published. A push to the development repository's default branch publishes. The
public repository is named in `declarations.json`.

## What ships

`declarations.json` declares every tracked path. `ships` and `private` list path prefixes, and a path takes the
longest entry it falls under, so a directory can ship while one file inside it stays private. A tracked path under
no entry refuses the export, and `mirror.py --check` refuses it at commit time: a new file or directory never ships,
and never silently stays behind, until someone declares it. The list names what ships rather than what hides,
because a list of hidden things fails open on the file nobody anticipated.

## The tripwire

The export is HEAD's shipping files, written by `git archive` (committed bytes only, never a build product). It is
refused if any file

- is an agent or scratch file at any depth (`CLAUDE.md`, `.claude/`, `SCRATCHPAD.md`), or
- holds a private-shaped string: a home directory, the private scratch repository's name, a personal email address,
  a cloud project or bucket identifier, or a credential (GitHub, Hugging Face, Anthropic and AWS token shapes, a
  private key block).

`allow` in `declarations.json` names shapes this repository's published files may carry, each with the reason. A
credential can never be allowed. Every pattern is spelled so that `mirror.py`'s own text does not match it. A
refusal prints each finding and exits 1, which fails the workflow before anything is pushed.

## The snapshot

`.github/workflows/mirror.yml` runs the export on the pushed commit, commits the export on top of the public
repository's `main` and pushes. Each publish is one dated commit naming the development commit it was taken from,
so the public history records what was published when. The push authenticates with the development repository's
`MIRROR_KEY` secret, the private half of a write deploy key on the public repository.

`mirror.py`, this README and the workflow are the same in every repository that mirrors; only `declarations.json`
differs.

```bash
python .github/mirror/mirror.py /tmp/export   # the export of HEAD, with any refusal
python .github/mirror/mirror.py --check       # every tracked path declared
```
