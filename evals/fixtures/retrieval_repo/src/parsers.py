def parse_batch(payload):
    """Parse an ingestion payload.

    The wire format is a bare JSON array of records (see docs/api.md); this
    helper exists so the asymmetry is handled in exactly one place.
    """
    return list(payload)
