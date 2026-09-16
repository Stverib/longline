# Pipeline notes

Working notes for the fan-out pipeline. Statements here are decisions that were
already agreed — reading the code will not tell you *why*, only *what*.

## Decisions

- **Chunking is sequential, not round-robin.** `chunk_tasks` returns
  consecutive slices. Round-robin was tried first and rejected because
  `merge_results` reconstructs the original order from the batch order, and
  interleaving made that impossible without carrying an index alongside every
  result.
- **`merge_results` must not deduplicate.** A key appearing in two batches is
  legitimate — it happens on retry, and at the boundary when a caller re-issues
  the tail of a batch. Dropping the repeat silently loses a real result.

## Open

`merge_results` is accumulating into a dict instead of a list, which is why the
frozen test is red. See the README for the current state.
