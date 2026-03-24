Docxology sidecar
=================

The repository includes **docxology/** — manifests, pytest, and markdown indexes that align runnable **examples/** notebooks and scripts with upstream **test/** coverage. It does not replace this Sphinx build; it complements it for operators and CI-style validation.

Repository paths (clone-relative):

* ``docxology/README.md`` — setup, ``uv`` groups, commands
* ``docxology/AGENTS.md`` — manifests, path roots, orchestration
* ``docxology/docs/examples_catalog.md`` — gallery tiers, deps, Sphinx notebook cross-links
* ``docxology/docs/validation_matrix.md`` — capability areas → ``test/`` → examples → these RST pages

**Notebooks:** nbval manifests under ``docxology/manifests/`` point at **``examples/**/*.ipynb``**. Tutorials under ``docs/notebooks/`` are built here with MyST-NB; they may differ from gallery copies for the same topic. Prefer **examples/** paths when matching CI manifests.

**Paper:** theoretical background for the library: `JOSS article <https://joss.theoj.org/papers/10.21105/joss.04098>`_.
