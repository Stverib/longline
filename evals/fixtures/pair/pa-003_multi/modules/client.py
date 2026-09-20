"""The client module.

Policy: A server that fails to connect is retried once, then disabled.
"""


def client_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'MCP clients are started lazily on first tool use.'
