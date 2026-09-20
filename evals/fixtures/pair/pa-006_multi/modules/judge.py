"""The judge module.

Policy: A judge that raises counts as failed, not as an abort.
"""


def judge_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Judges are deterministic checks over a sandbox.'
