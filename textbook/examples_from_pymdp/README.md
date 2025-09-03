# examples_from_pymdp

This directory provides standalone, executable Python scripts that mirror the example notebooks in `examples/` from this repository. Scripts are generated so that each `.py` contains exactly the concatenation of all code cells from the corresponding `.ipynb` in the same order, without any added headers, comments, or extra lines. Where an example `.py` already exists (e.g., `examples/agent_demo.py`), it is copied verbatim.

## Contents

- `generate_examples_from_pymdp.py`: Converts listed notebooks into `.py` scripts by extracting code cells in order; copies any existing `.py` examples as-is.
- `run_all_pymdp_examples.py`: Runs the generator, then executes all generated example scripts.
- Generated scripts (created on demand):
  - `agent_demo.py` (copied from `examples/agent_demo.py`)
  - `building_up_agent_loop.py` (from `examples/building_up_agent_loop.ipynb`)
  - `free_energy_calculation.py` (from `examples/free_energy_calculation.ipynb`)
  - `gridworld_tutorial_1.py` (from `examples/gridworld_tutorial_1.ipynb`)
  - `gridworld_tutorial_2.py` (from `examples/gridworld_tutorial_2.ipynb`)
  - `inductive_inference_example.py` (from `examples/inductive_inference_example.ipynb`)
  - `inductive_inference_gridworld.py` (from `examples/inductive_inference_gridworld.ipynb`)
  - `model_inversion.py` (from `examples/model_inversion.ipynb`)
  - `testing_large_latent_spaces.py` (from `examples/testing_large_latent_spaces.ipynb`)
  - `tmaze_demo.py` (from `examples/tmaze_demo.ipynb`)
  - `tmaze_learning_demo.py` (from `examples/tmaze_learning_demo.ipynb`)

Notes:
- If both `examples/<name>.py` and `examples/<name>.ipynb` exist with the same basename, the existing `.py` is copied and the `.ipynb` is skipped to avoid collision. This preserves the original `.py` exactly.
- Generated `.py` scripts contain only the code from notebook cells; markdown, outputs, and metadata are omitted by design.

## Usage

From the repository root:

```bash
python3 textbook/examples_from_pymdp/run_all_pymdp_examples.py
```

This will:
1) Generate all scripts under `textbook/examples_from_pymdp/`.
2) Execute each script in an isolated harness that:
   - switches CWD to `textbook/examples_from_pymdp/outputs/<example_name>`
   - uses a non-interactive matplotlib backend and auto-saves any open figures to that folder
   - captures all relative file outputs into that folder
3) Summarize successes and failures.

Alternatively, to only (re)generate scripts without running them:

```bash
python3 textbook/examples_from_pymdp/generate_examples_from_pymdp.py
```

You can then run any individual generated script, for example:

```bash
python3 textbook/examples_from_pymdp/gridworld_tutorial_1.py
```

## Guarantees

- Exact code ordering: For notebooks, the output `.py` is the exact concatenation of all code cells in order.
- No extraneous content: No boilerplate, headers, or extra lines are added by the generator.
- Verbatim copies: Existing example `.py` files are copied byte-for-byte.
- Per-example output folders: Running via the provided runner saves all figures and any relative outputs into `textbook/examples_from_pymdp/outputs/<example_name>/`.

## Requirements

- Python 3.7+ (matches notebooks and repository support)
- Any runtime dependencies used within the examples themselves (e.g., numpy, matplotlib, jax, seaborn, etc.). Install via the repository requirements if needed:

```bash
pip install -r requirements.txt
# or for docs examples
pip install -r docs/requirements.txt
```

## Re-generating after edits

If you update any notebook under `examples/`, re-run the generator to refresh the corresponding `.py` script.

```bash
python3 textbook/examples_from_pymdp/generate_examples_from_pymdp.py
```

## Caveats

- Some notebooks rely on interactive environments (e.g., plotting windows). When executed as scripts, they will run the same code; behavior (e.g., blocking windows) depends on your environment and matplotlib backend.
- JAX notebooks assume a JAX-enabled environment; running those scripts requires the appropriate JAX installation.

