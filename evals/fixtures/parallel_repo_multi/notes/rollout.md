# Rollout checklist

Walk these in order. Each section belongs to one module and can be completed
independently of the others.

## Routing

- [ ] `python -m src.router --check` exits 0
- [ ] `/health` resolves without touching the cache

## Cache

- [ ] `DEFAULT_TTL_S` matches the value in the deploy config
- [ ] The eviction path has been exercised with `MAX_ENTRIES = 2`

## Retry

- [ ] `MAX_ATTEMPTS` is <= 4
- [ ] A 429 is retried and a 400 is not

## Sign-off

- [ ] The four modules above are all listed as checked
