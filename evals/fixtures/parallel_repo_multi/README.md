# svc

A small service split into four independent modules.

## Layout

| path | role |
|---|---|
| `src/router.py` | request routing |
| `modules/cache.py` | response cache |
| `modules/retry.py` | retry policy |
| `notes/rollout.md` | rollout checklist |

Each module is self-contained: none imports another, and none is imported by
`src/router.py`. They were split apart precisely so a change to one cannot
affect the others.

## How it ships

```bash
python -m src.router --check
```

`notes/rollout.md` is the checklist an operator walks before a deploy; the four
sections below list what each module is expected to say for itself.
