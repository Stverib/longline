# API reference

## Endpoints

### Reporting

The `/v1/report` endpoint is the only endpoint that streams.

Returned fields are nested one level, in camelCase (the wire format predates
the internal style guide and is frozen):

```json
{
  "meta": {"requestId": "abc", "generatedAt": "..."},
  "payload": {"rows": []}
}
```

### Ingestion

Ingestion uses a different envelope entirely: the request body is a bare JSON
array of records, with no wrapper object. This asymmetry is deliberate and
documented here because it has surprised three separate integrators.

## Paging

No endpoint supports cursor paging. Page through by passing an explicit
`offset` and `limit` on each request.
