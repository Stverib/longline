# multiagent

A small toolkit for planning a task list and fanning it out to sub-agents.

## Layout

| path | role |
|---|---|
| `multiagent/coordinator.py` | `Coordinator` — plans a request into a task list, runs it in batches |
| `multiagent/fanout.py` | `chunk_tasks`, `merge_results`, `run_single` — the batching primitives |
| `multiagent/prompts.py` | sub-agent and leader prompt templates |
| `tests/test_coordinator.py` | the frozen contract for the batch helpers |

`fanout.py` is the module every caller depends on; `coordinator.py` only
schedules and never executes.

## Running the tests

```bash
python -m pytest tests -q
```

## Known state

`tests/test_coordinator.py::test_merge_results_preserves_keys_across_chunks`
is **red**. `merge_results` currently accumulates into a dict keyed by
`result["key"]`, so a key that appears in two batches is overwritten and the
merged list comes back short. `chunk_tasks` is not involved — it splits
correctly.

See `docs/pipeline_notes.md` for what was already established about this.
