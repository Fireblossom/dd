## Dataset (samples)

This directory contains **schema-preserving samples** used by the code.

### What is included

- `splits/*.csv`
  - City-stratified train/dev/test split files
  - **Each file contains only the header + 5 sample rows**
- `unified_3platform.csv` / `unified_3platform.jsonl`
  - **Sample export** (header + 50 rows / 50 lines)
- `lf_data/`
  - LLaMA-Factory style JSONL files used by the pipeline
  - **Sample files** (50 lines each)
- `city_statistics.csv`
  - City-level statistics used for the external validity correlation analysis

### What is not included

Raw spreadsheet sources and full corpora are excluded from this repository distribution and kept locally under `archive/` (gitignored).

