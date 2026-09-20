"""The session module.

Policy: Session writes are atomic; a partial session is never visible.
"""


def session_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Sessions are addressed by id and stored under the config dir.'
