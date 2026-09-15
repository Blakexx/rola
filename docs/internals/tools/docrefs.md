# `tools/docrefs.py` — the doc references

Mirrors `tools/docrefs.py`. Every citation a doc makes of the tree is pinned to the content of what it cites, so a
change to that content marks the citation for rereading instead of leaving it to rot. Docs are rewritten in one pass
at the end of a piece of work, with the whole change in view, so a changed target does not fail a commit; it waits in
`pending` until that pass.

## What a reference is

`references` reads every tracked markdown file except the deletion ledger (a ledger row describes the tree at its own
commit), outside fenced code blocks, and takes four forms:

- a possessive, a path in code font followed by `'s` and a name in code font: the name's primitive in that file;
- a dotted name in code font whose first component is one of `PACKAGES`: the longest prefix that is a module in the
  tree, and the rest as the primitive in it;
- a path in code font with a known suffix or a trailing `/`: the file or directory;
- a markdown link: the file, or the section its `#anchor` names.

`locate` finds a target in a `Version` of the tree (the working tree, or a commit). A Python primitive is a function,
class, method or constant, found through `ast` and followed through a `from x import name` re-export; it is hashed as
`ast.unparse` of the node with docstrings dropped, so a comment, a docstring or a reformat moves nothing. A device or
host C++ primitive is a `//: @ref NAME` ... `//: @end` region, else the first brace-matched definition of the name,
hashed with comments dropped and whitespace collapsed; a name the file carries without either is checked only to
exist. A markdown section runs from its heading (or `<a id>`) to the next heading at its level or above. A file or a
directory is checked only to exist, and a build output the tree ignores (`_generated`) counts as existing.

## The locks

`docs/references.lock.json` holds one pin per (doc, target): the digest, the commit it was pinned at, and whether a
docs pass has read it. A doc the public mirror does not ship keeps its pins in `.claude/references.lock.json`
(`_visibility` reads `.github/mirror/declarations.json`), so the public lock never names it. `init` pinned the tree's
references unread, the broken ones with a null digest.

## The gate and the pass

`verdict` compares the survey to the locks. The gate (`python3 tools/docrefs.py`, a pre-commit hook and a CI step)
fails on a reference no pin holds (read it, then `accept` its doc; a reference that resolves to nothing cannot be
accepted), a pin whose reference no doc makes any more (`prune`), and a broken pin HEAD's lock did not hold, so the
broken backlog only shrinks. A pinned reference whose target's digest moved or vanished is pending: `pending` prints
each target with the docs citing it and the target's own source as a diff from its pin (`change`).

`accept DOC` pins the doc's new and changed references; `accept --whole DOC` also marks every other reference in it
read, for the docs pass that read the whole page. `--closing` is the pass's end: it fails on anything pending, unread
or broken, and lists each.
