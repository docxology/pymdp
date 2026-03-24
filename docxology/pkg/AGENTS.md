# docxology/pkg

## Layout

- **`__init__.py`** — Package marker for `pkg` imports from `docxology/examples/…`.
- **`support/`** — Implementation helpers for mirror scripts; not a published PyPI API.

## Import pattern

Mirror scripts run with `cwd` = `docxology/` and typically import `pkg.support.*` after bootstrap adjusts `sys.path` ([support/bootstrap.py](support/bootstrap.py)).

## Parent

[../AGENTS.md](../AGENTS.md)
