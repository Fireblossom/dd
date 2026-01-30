## Reproducibility & artifacts

This repository contains:

- **Paper source**: `paper/` (entry: `paper/main.tex`)
- **Code**: `src/`
- **Data (schema-preserving samples)**: `dataset/`
- **Human-readable outputs**: `outputs/` and `paper/figures/`

### Data note

Some raw sources (e.g., platform spreadsheets) and the full text corpus are not included here due to access restrictions. The `dataset/` directory contains small samples so the pipeline can run and the schemas are fully visible.

### Where results live

- Figures referenced in the paper: `paper/figures/`
- City profile report + plots: `outputs/city_profiles/`
- Cross-platform report (precomputed): `outputs/cross_platform/`
- External validity / correlations: `outputs/icwsm/external_validity/`

### Run the pipeline (on included samples)

```bash
uv sync
uv run python src/run_experiments.py
```

Run a single experiment:

```bash
uv run python src/run_experiments.py -e 1
uv run python src/run_experiments.py -e 2
uv run python src/run_experiments.py -e 3
uv run python src/run_experiments.py -e 4
```

Cross-platform (Douyin vs WeChat) requires raw spreadsheet sources; if those files are absent, the runner skips that step and you can read the precomputed report under `outputs/cross_platform/`.

