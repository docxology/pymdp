# Building the Docs

## Local preview

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

## Build static site

```bash
mkdocs build
```

If Mermaid clicks do not work, verify mkdocs.yml has the Mermaid plugin and loose security enabled.
