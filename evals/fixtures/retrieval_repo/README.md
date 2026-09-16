# retrieval_repo

A small repository used to measure whether a task can be answered from what is
actually in the tree, rather than from what a model remembers.

The answers to the retrieval cases are spread across `docs/`, `CHANGELOG.md`,
`src/` and `data/`, and several of them deliberately contradict the obvious
guess — one changelog entry records a flag being *removed*, one endpoint
returns a different envelope from the others, and one handler is present but
unrouted.

Nothing in this repository is generated; every file is the starting state.
