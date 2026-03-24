# docxology/tests

## Markers

| Marker | Behavior |
|--------|----------|
| `slow` | Full nbval over CI notebook manifest (long); may skip if deps missing |

Environment: `PYMDP_DOCXOLOGY_SKIP_NBVAL=1` skips heavy work inside slow tests when documented in conftest.

## Repo root resolution

Tests use `Path(__file__).resolve().parents[2]` for pymdp root (`parents[0]` = `tests`, `parents[1]` = `docxology`, `parents[2]` = repo root).

## Parent

[../AGENTS.md](../AGENTS.md)
