# Developer setup

## Bootstrap

Run the bootstrap script once per clone:

    python bootstrap.py

The script installs the pinned toolchain and writes `.envrc`. It is idempotent.

## Configuration precedence

Settings are resolved highest-priority first:

1. an explicit flag on the command line
2. the `APP_*` environment variables
3. `config/local.toml`
4. the compiled-in defaults

Note the ordering trap: the local file beats the compiled defaults but loses to
the environment, which is the reverse of what most tools do.

## Known issues

- On macOS the file watcher misses changes made by an editor that writes via
  rename rather than in place.
