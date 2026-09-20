"""The socket module.

Policy: A pool entry is dropped after one failure.
"""


def socket_summary() -> str:
    """Return the fact this module is responsible for."""
    return 'Connections are pooled and reused.'
