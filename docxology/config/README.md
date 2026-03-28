# docxology/config

> **Orchestration Context & Rigor Settings**

Optional YAML defaults for docxology runners. These configurations allow you to toggle between fast smoke tests and rigorous full-depth validations.

| File | Consumer | Description |
| :--- | :--- | :--- |
| [`orchestration.yaml`](orchestration.yaml) | [`scripts/run_docxology_orchestrations.py`](../scripts/run_docxology_orchestrations.py) | Global defaults for `seed`, `fast`, `skip_heavy`, and `verbose`. |

### Key Features
- **Deterministic Validation**: Set a global `seed` to ensure reproducible agent trajectories.
- **CI/CD Optimization**: Toggle `fast: true` to execute minimal timesteps for rapid pipeline verification.
- **Selective Resource Management**: Use `skip_heavy: true` to bypass deep-tree MCTS or SI scenarios in resource-constrained environments.

> [!NOTE]
> Command-line arguments always override YAML settings. Requires `pyyaml` (docxology `test` dependency group).

See [AGENTS.md](AGENTS.md) for the machine-readable contract.
