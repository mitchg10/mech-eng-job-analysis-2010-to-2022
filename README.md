# Mechanical Engineering Job Market Analysis (2010-2022)

## Project Overview

This repository contains code and methodology for analyzing mechanical engineering job market trends from 2010 to 2022 using Burning Glass Technologies (BGT) labor market data. This research was sponsored by the National Science Foundation (NSF).

The analysis pipeline processes approximately 1.17 million mechanical engineering job postings extracted from a larger dataset of 422 million total jobs spanning 2007-2023. The methodology includes:

- NLP-based classification of mechanical-major vs. non-mechanical-major positions
- Temporal trend analysis for 200-250 common skills with prevalence ≥2%
- Statistical modeling with confidence tiers (Strong/Moderate/Weak/Exploratory)
- IEEE-format publication tables

The pipeline consists of three sequential phases:

1. **Phase 1: Data Extraction** - Filter and extract mechanical engineering jobs from BGT dataset
2. **Phase 2: NLP Classification** - Classify jobs using Bayesian-optimized confidence scoring
3. **Phase 3: Skills Analysis** - Extract skills, model temporal trends, generate publication outputs

## Data Availability Notice

**Important: This repository contains code and methodology only. The Burning Glass Technologies dataset is not included and is not publicly available.**

The BGT data is proprietary labor market intelligence provided by Burning Glass Technologies (now Lightcast). Researchers interested in accessing this data should contact Lightcast directly at [www.lightcast.io](https://www.lightcast.io).

The code in this repository demonstrates the analytical methodology and can serve as a reference for researchers with access to similar labor market datasets.

## Environment Setup

### System Requirements

- **Python Version:** >=3.13
- **Internal Dependency:** DuckDB Manager module (required for Phase 1 only)

### Installation

Install the package and its dependencies using pip or uv:

```bash
# Using pip
pip install -e .

# Using uv (faster alternative)
uv pip install -e .
```

### Key Dependencies

The following packages are automatically installed via `pyproject.toml`:

**Data Processing:**
- polars (>=1.38.0) - Primary data processing framework
- pandas (>=3.0.0) - Used for statsmodels compatibility
- pyarrow (>=23.0.0) - Parquet file I/O
- duckdb (>=1.4.4) - Query engine for BGT dataset

**Machine Learning & NLP:**
- torch (>=2.10.0) - Deep learning framework
- transformers (>=5.0.0) - Transformer models
- sentence-transformers (>=5.2.2) - Sentence embeddings
- scikit-learn (>=1.8.0) - Machine learning utilities

**Statistical Modeling:**
- statsmodels (>=0.14.6) - Regression and time series models
- optuna (>=4.7.0) - Bayesian hyperparameter optimization

**Visualization:**
- matplotlib (>=3.10.8) - Plotting library
- seaborn (>=0.13.2) - Statistical visualizations
- plotly (>=6.5.2) - Interactive plots

**Development:**
- pytest (>=9.0.2) - Testing framework
- ipykernel (>=7.1.0) - Jupyter notebook support

## File Structure

### Notebooks

- `phase_1.ipynb` - Data extraction from BGT dataset 
- `phase_2.ipynb` - NLP classification of job postings 
- `phase_3.ipynb` - Skills extraction and temporal trend analysis

### Configuration Files (`config/`)

- `analysis_config.json` - Pipeline parameters (study period, paths, thresholds, chunk sizes)
- `mechanical_terms.json` - Classification vocabulary (O*NET codes, core terms, software tools, industry context)
- `classifier_parameters.json` - Bayesian-optimized weights and thresholds (150 trials, TPE sampler)

### Utility Modules (`utils/`)

- `onet_utils.py` - O*NET code format conversion and discipline mapping
- `classifier_utils.py` - NLP job classification with multi-signal confidence scoring
- `skills_utils.py` - Skills parsing (pipe-delimited format) and prevalence calculations
- `statistics_utils.py` - Temporal trend modeling (AICc selection, diagnostics, confidence tiers)
- `visualization_utils.py` - Trajectory plots and residual diagnostics
- `publication_utils.py` - IEEE-format table generation (8 tables: CSV + LaTeX)

## Expected Outputs

When run with appropriate data access, the pipeline generates the following outputs in the `{base_path}/Analysis/` directory:

### Parquet Files

- `study_data_ME_ONLY.parquet` - Extracted mechanical engineering jobs (~1.17M records)
- `classified_ME_ONLY.parquet` - Jobs with classification labels (~1.17M records)
- `skills_ME_ONLY.parquet` - Long-format skill-job pairs (~9M records)

### CSV Summaries

- `skill_trajectory_summary.csv` - Temporal trends for 200-250 common skills
- `regression_diagnostics.csv` - Statistical diagnostic test results (Shapiro-Wilk, Breusch-Pagan, Durbin-Watson)

### Publication Tables (`ieee_tables/`)

Eight IEEE-format tables (both CSV and LaTeX):
1. Top growing skills (Tier 1-2)
2. Top declining skills (Tier 1-2)
3. Stable high-prevalence skills
4. High-demand skills
5. Skills by model type (linear, log, exponential, quadratic)
6. Tier 3 growth trajectories
7. Tier 3 decline trajectories
8. All skills summary

### Visualizations (`plots/`)

- Trajectory grid plots by confidence tier (Tier 1, 2, 3)
- Residual diagnostic plots per skill

## Pipeline Execution

### Sequential Execution Required

Each phase depends on outputs from the previous phase. Execute notebooks in order:

```bash
# Phase 1: Data Extraction
jupyter notebook phase_1.ipynb
# Expected output: ~1.17M mechanical engineering jobs
# Runtime: ~10-20 minutes (DuckDB query + Parquet export)

# Phase 2: NLP Classification
jupyter notebook phase_2.ipynb
# Expected output: ~84% mechanical-major, ~16% non-mechanical-major
# Runtime: ~40-50 minutes (50k chunk processing with progress updates)

# Phase 3: Skills Analysis
jupyter notebook phase_3.ipynb
# Expected output: ~200-250 common skills, ~9M skill-job pairs
# Runtime: ~20-30 minutes (1-3 seconds per skill model fitting)
```

### Cell-by-Cell Execution

Notebooks include:
- Markdown documentation explaining each step
- Diagnostic outputs and summary statistics
- Progress updates (every 50k records in Phase 2, every 10% in Phase 3)
- Caching mechanisms (Phase 3 checks for cached skills parsing)

### Key Checkpoints

Verify expected results at each phase:

- **Phase 1:** ~1.17M jobs extracted for study period (2010-2022)
- **Phase 2:** Classification split approximately 84% mechanical-major / 16% non-mechanical-major
- **Phase 3:** ~200-250 skills meeting ≥2% average prevalence threshold, ~9M skill-job pairs in long format

## Configuration Updates

### Study Period

Edit `config/analysis_config.json` → `phase_1.study_period`:

```json
{
  "start": "2010-01-01",
  "end": "2022-12-31"
}
```

### Classification Sensitivity

Edit `config/classifier_parameters.json` → `adaptive_thresholds`:

```json
{
  "standard": 3.9,        // Lower = more inclusive classification
  "definitive_positive": 7.26
}
```

### Skill Prevalence Threshold

Edit `config/analysis_config.json` → `phase_3.min_prevalence`:

```json
{
  "min_prevalence": 0.02  // 2% threshold yields ~200-250 skills
}
```

## Methodology Notes

### BGT Skill Cluster Format

Skills are stored in pipe-delimited format: `"Family: Skill; Type | Family: Skill; Type"`

- Complete entries: "Engineering: CAD; Technical Skills"
- Missing type: Uses "Specialized Skills" default
- Type-only entries: Skipped (no family/skill name)

### Statistical Approach

Due to small sample size (n=13 years):
- AICc correction used instead of AIC (finite-sample bias correction)
- Diagnostic tests have limited power
- Confidence tiers (1-4) provide transparency about evidence strength
- Tier 4 (Exploratory) should not be used for strong conclusions

Model selection includes parsimony principle:
- If Δi < 2 (models are equivalent), prefer simpler model
- Null model preferred when evidence is weak
- Prevents overfitting with small sample size

### O*NET Code Formats

Two representations used:
- **Standard format:** `'17-2141.00'` (human-readable)
- **Numeric format:** `'17214100'` (used in BGT dataset)

Convert using `onet_utils.convert_onet_standard_to_numeric()` and `onet_utils.convert_onet_numeric_to_standard()`.

## Citation

If you use this methodology in your research, please cite:

```
[Citation information to be added upon publication]
```

## License

[License information to be added]

## Contact

For questions about the methodology or code, please open an issue on GitHub.

For questions about accessing Burning Glass Technologies data, contact Lightcast at [www.lightcast.io](https://www.lightcast.io).
