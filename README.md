# Rossmann Store Sales Forecasting: Hand-Engineered LightGBM vs AutoML

> A hand-engineered LightGBM pipeline with 50+ features beat the LightAutoML (LAMA) baseline by ~11% on Kaggle's Rossmann Store Sales leaderboard, reaching **0.124 RMSPE (private)** against **0.139** — where the simpler feature set generalised better than the expanded one.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Feature%20Based-4E8CBE)](https://github.com/microsoft/LightGBM)
[![LightAutoML](https://img.shields.io/badge/LightAutoML-Baseline-6B7A8F)](https://github.com/AILab-MLTools/LightAutoML)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-E6A817)](https://optuna.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Why this exists

Retail forecasting at store level is dominated by strong weekly seasonality, promotion effects and long-tail store behaviour. The reference baseline from LightAutoML (LAMA) already performed well, but AutoML leaves feature intelligence on the table: the model never sees engineered interactions like store-day means or promo-dow effects. The question this project answers is whether deliberate, hypothesis-driven feature engineering can beat an out-of-the-box AutoML pipeline on a fixed Kaggle task.

The approach was to run the EDA first, turn each insight into a testable hypothesis, build custom sklearn transformers that encode those hypotheses, and validate every step against a time-based split. Instead of adding complexity blindly, the project also measured what to **remove**: the final model dropped several engineered features and generalised better as a result.

The headline result: **0.124 RMSPE (private)** with the base LightGBM pipeline vs **0.139** for the best LightAutoML baseline — an ~11% relative error reduction, achieved with a fraction of LAMA's training time.

## Optimization & impact

| What | Before | After | How |
|------|--------|-------|-----|
| Forecast error (RMSPE, private) | 0.139 (LightAutoML LAMA) | **0.124** | hypothesis-driven feature engineering + LightGBM |
| Forecast error (RMSPE, public) | 0.126 (LAMA baseline 1) | **0.121** | feature subset selection (simpler generalises better) |
| Feature set vs generalisation | 0.129 (expanded feature set) | **0.124** (base set) | aggressive pruning of weak/leaky features |
| Feature search space | 16 hypotheses | 12 confirmed / 4 rejected | evidence-gated feature design |

Numbers taken from the validation runs and Kaggle submissions in `submissions/` and the research notebook. The 0.124 vs 0.129 result is on private, i.e. keeping the base feature set beat the expanded one.

## How it works

- Full research pipeline in a single notebook: `Rossman full research.ipynb` — EDA, hypothesis definition, feature engineering, baseline comparison, final models and ablation.
- Custom sklearn transformers implement the hypothesis layer: `StoreDowMean`, `MonthRatio`, `PromoDow`, `SalesRatio`, `WeekDowMean`, `LogCompDist` and interaction flags are computed as pipeline steps, then dropped or kept based on measured impact.
- Three LightAutoML baselines (`submissions/lama_*.csv`, `submission_lama*.csv`) establish the reference RMSPE; LightGBM variants (`lightgbm_base.csv`, `lightgbm_optuna.csv`, `lightgbm_autoregressive.csv`, `lightgbm_target_encoding*.csv`) isolate the effect of each feature block.
- Optuna tunes LightGBM hyperparameters (`feature_fraction`, `num_leaves`, `bagging_fraction`, `reg_alpha/lambda`); the target is trained on `log1p(Sales)` because RMSPE penalises relative error.

Key component files:
- `Rossman full research.ipynb` — the complete experiment log
- `src/submitsScreen.png` — Kaggle leaderboard screenshot
- `src/timeSeriesPrediction.png` — per-store validation forecasts
- `submissions/` — every submission with its feature configuration

## Results

| Model | RMSPE (val) | RMSPE (public) | RMSPE (private) |
|-------|------------|----------------|-----------------|
| LightAutoML LAMA baseline 1 | 0.1346 | 0.126 | 0.139 |
| LightAutoML LAMA baseline 2 | — | 0.140 | 0.149 |
| LightAutoML LAMA baseline 3 | — | 0.147 | 0.157 |
| **LightGBM, base features** | 0.137 | **0.121** | **0.124** |
| LightGBM, expanded features | 0.130 | — | 0.129 |

Top-5 features by gain importance in the final model:

| Feature | Importance | Why it works |
|---------|-----------|--------------|
| `StoreDowMean` | 755k | per-store weekly pattern (EDA correlation 0.84) |
| `MonthRatio` | 131k | deviation of a store from its type-month norm |
| `PromoDow` | 114k | promo interaction with day of week |
| `Store` | 74k | store identity encodes unique behaviour |
| `WeekDowMean` | 58k | micro-seasonality of specific weeks of the year |

What failed (and why it was removed): `NewCompetition` (zero importance), `CompetitionOpen` duration (1.6x weaker than `LogCompDist`), per-month `Promo_*` flags (importance < 150), binary combo flags (`DangerousComp`/`SafeComp`, never reached top-30), target lags (error accumulates over the forecast horizon) and per-store-type specialised models (each subset too small to train well). These rejections are documented in the notebook's final conclusions.

![Kaggle submissions](src/submitsScreen.png)

![Per-store validation forecasts](src/timeSeriesPrediction.png)

## Tech stack

| Category | Technology |
|----------|------------|
| Language | Python 3.12+ |
| Modelling | LightGBM, LightAutoML (LAMA) |
| Feature engineering | custom sklearn `BaseEstimator`/`TransformerMixin` pipeline |
| Hyperparameter tuning | Optuna |
| Data | Kaggle Rossmann Store Sales — 1.017M rows, 1,115 stores, 22 raw features |

## Quick start

```bash
uv sync
```

Then run `Rossman full research.ipynb` end-to-end (EDA, baselines, feature engineering, final models). Data files go in `data/` (`train.csv`, `test.csv`, `store.csv`, `sample_submission.csv`). All models and submission CSVs in `submissions/` are reproducible from the notebook.

## Project structure

```
.
├── Rossman full research.ipynb   # full research pipeline
├── data/                         # Kaggle raw data
├── submissions/                  # baseline + feature-ablation submissions
└── src/
    ├── submitsScreen.png         # Kaggle leaderboard screenshot
    └── timeSeriesPrediction.png  # sample forecasts for 5 stores
```

## Acknowledgements

Dataset: Kaggle Rossmann Store Sales. Baseline framework: LightAutoML. Metrics and leaderboard evaluation follow the competition's RMSPE definition.

## License

MIT License. See [LICENSE](LICENSE).
