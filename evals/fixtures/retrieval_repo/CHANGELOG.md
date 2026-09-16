# Release notes

Each entry records the ticket that closed it. Retired tickets are kept here on
purpose: grepping the current source for an old number finds nothing.

- `RET-4021` — removed the deprecated `--legacy-parser` flag. The flag had been
  a no-op since the migration, so no compatibility shim was left behind.
- `RET-4033` — bumped the default request timeout from 30s to 45s.
- `RET-4050` — replaced the in-memory response cache with a filesystem cache
  under `.cache/`.
- `RET-4062` — retired the `SIGNING_MODE` environment variable. Signing is now
  always required.
- `RET-4077` — the `/v1/batch` endpoint was renamed to `/v1/bulk`. The old
  route was removed rather than aliased.
