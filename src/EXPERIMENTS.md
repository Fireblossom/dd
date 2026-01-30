# Urban Identity Analysis - Experiments

Reproducible commands for analyzing how urban identity is constructed across Chinese megacities and social media platforms.

## Quick Start

```bash
# Run all experiments
uv run python3 src/run_experiments.py

# Run individual experiments
uv run python3 src/run_experiments.py -e 1   # City Profiles
uv run python3 src/run_experiments.py -e 2   # Cross-Platform
uv run python3 src/run_experiments.py -e 3   # Socioeconomic Correlation
uv run python3 src/run_experiments.py -e 4   # LLM Reliability
```

---

## Data

| Platform | File | Samples | Cities |
|----------|------|---------|--------|
| Douyin | `dataset/抖音.xlsx` | 411 | Beijing, Shanghai, Guangzhou, Shenzhen |
| WeChat Channels | `dataset/微信视频号.xlsx` | 386 | Beijing, Shanghai, Guangzhou, Shenzhen |
| Xiaohongshu | `dataset/lf_data/*.jsonl` | ~2000 | 14 cities (filter to 4) |

**Data availability note:** Some raw sources and full corpora are not included here due to access restrictions.
The `dataset/` directory contains small **schema-preserving samples** so the pipeline can be executed end-to-end.

---

## Experiments

### 1. City Profiles
Analyze identity construction patterns per city.

```bash
uv run python3 src/city_profile_analysis.py
```

**Outputs:**
- `outputs/city_profiles/city_profiles_report.md`
- `outputs/city_profiles/city_criteria_heatmap.png`
- `outputs/city_profiles/city_radar_chart.png`

### 2. Cross-Platform Comparison
Compare discourse patterns across Douyin vs WeChat.

```bash
uv run python3 src/cross_platform_analysis.py
```

**Outputs:**
- `outputs/cross_platform/cross_platform_report.md`
- `outputs/cross_platform/combined_data.csv`

### 3. Socioeconomic Correlation
Correlate identity patterns with city statistics.

```bash
uv run python3 src/external_validity.py \
  --stats_file dataset/city_statistics.csv \
  --out_dir outputs/socioeconomic
```

**Required:** Fill in `dataset/city_statistics.csv` with:
- `migrant_ratio` - Non-hukou population ratio
- `house_price_ratio` - Price-to-income ratio
- `dialect_usage` - Local dialect usage rate

### 4. LLM Reliability Testing
Test if LLMs can reliably identify identity dimensions.

```bash
uv run python3 src/prompt_sensitivity.py \
  --data_dir dataset/lf_data \
  --output_dir outputs/llm_reliability
```

---

## Environment

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
uv sync
```

## Key Findings (Preliminary)

**City Differences:**
- Shenzhen: Family Heritage (43%) + Property (27%) - emphasizes lineage and assets
- Shanghai: Language (22%) + Intra-city (23%) - dialect and district identity
- Guangzhou: Family Heritage (26%) + Language (22%) - Cantonese culture
- Beijing: More dispersed patterns

**Platform Differences:**
- Douyin: More emphasis on Family Heritage (38% vs 18%)
- Douyin: More restrictive attitudes (39% vs 32%)
