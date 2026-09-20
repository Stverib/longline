"""The mailbox module.

Policy: Reading an inbox does not consume it; marking does.
"""


def mailbox_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Each agent owns one inbox file holding a message list.'
