"""Dispatch table for the sample service.

The table lives here rather than in a config file so that the tests can import
it directly.
"""

HANDLERS = {
    "/v1/report": "report_handler",
    "/v1/bulk": "bulk_handler",
    # The staged ingestion path is NOT wired up yet: the handler exists but no
    # production route points at it.
    "/v1/ingest/staged": None,
}


def timeout_seconds() -> int:
    """Per-request timeout applied by the client wrapper."""
    return 45
