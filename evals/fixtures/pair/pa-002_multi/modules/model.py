"""The model module.

Policy: A missing model id fails loudly instead of falling back.
"""


def model_describe() -> str:
    """Return the fact this module is responsible for."""
    return 'Models are addressed by id and resolved at call time.'
