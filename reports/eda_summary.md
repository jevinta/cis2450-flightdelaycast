# EDA Findings Summary

## 1) Data context and project objective
- Objective: predict whether arrival delay is greater than 15 minutes (`is_delayed`).
- Dataset size: **60,000** rows.
- Key feature groups: schedule (`dep_hour`, `month`, `day_of_week`), route (`ORIGIN`, `DEST`, `DISTANCE`), carrier (`OP_CARRIER`), optional weather (`w_*`, `dw_*`).
- Roadmap impact: confirms that this is a classification task with mixed numeric and categorical predictors.

## 2) Summary statistics (key numeric features)
| stat | dep_hour | month | day_of_week | DISTANCE | w_tmax | w_prcp |
|---|---|---|---|---|---|---|
| count | 60000.00 | 60000.00 | 60000.00 | 60000.00 | 5481.00 | 4779.00 |
| mean | 13.33 | 1.00 | 2.82 | 805.68 | 10.59 | 2.68 |
| std | 4.84 | 0.00 | 2.00 | 586.93 | 8.13 | 6.93 |
| min | 0.30 | 1.00 | 0.00 | 31.00 | -29.30 | 0.00 |
| 25% | 9.10 | 1.00 | 1.00 | 378.00 | 5.00 | 0.00 |
| 50% | 13.20 | 1.00 | 3.00 | 651.00 | 10.60 | 0.00 |
| 75% | 17.33 | 1.00 | 5.00 | 1046.00 | 16.10 | 1.50 |
| max | 23.98 | 1.00 | 6.00 | 5095.00 | 30.00 | 54.60 |

- `dep_hour`: fractional hour (0–23) parsed from BTS `CRS_DEP_TIME`; spread across the full day.
- `DISTANCE`: right-skewed (short hops dominate; long-haul flights are outliers handled by RobustScaler).
- Weather columns (`w_*`) have high missing rates — median imputation applied in the model pipeline.

## 3) Target distribution (class balance)
- Delay prevalence: **23.30%** delayed vs **76.70%** not delayed.
- Interpretation: class imbalance is present, so model evaluation should emphasize precision/recall/F1 (not accuracy alone).
- Roadmap impact: use imbalance-aware evaluation and keep probability outputs for threshold tuning.

## 4) Distribution and pattern checks
- Time-of-day effect: highest delay risk appears around hour **18:00** at **29.19%**.
- Carrier-level variation indicates airline identity contributes useful signal.
- Correlation matrix helps flag redundant numeric predictors before modeling.
- Roadmap impact: retain hour and carrier features; monitor multicollinearity among numeric variables.

## 5) Outlier check and handling rationale
- Method: IQR rule (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`) on key numeric variables.
- `w_prcp`: 782/4,779 potential outliers (16.36%).
- `DISTANCE`: 3,267/60,000 potential outliers (5.45%).
- `w_tmax`: 58/5,481 potential outliers (1.06%).
- `dep_hour`: 0/60,000 potential outliers (0.00%).
- Handling policy: keep valid operational extremes (e.g., long-distance flights, extreme weather) and rely on robust models/tree splits; add clipping only if validation metrics degrade.
- Roadmap impact: preserves real rare events while keeping a documented mitigation strategy.

## 6) EDA conclusion
- EDA supports the modeling approach: mixed-feature classification with interpretable risk outputs.
- Next steps: continue threshold tuning and error analysis by carrier, hour, and weather regime.
