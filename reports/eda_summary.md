# EDA Findings Summary

## 1) Data context and project objective
- Objective: predict whether arrival delay is greater than 15 minutes (`is_delayed`).
- Dataset size: **150,000** rows.
- Key feature groups: schedule (`dep_hour`, `month`, `day_of_week`), route (`ORIGIN`, `DEST`, `DISTANCE`), carrier (`OP_CARRIER`), optional weather (`w_*`, `dw_*`).
- Roadmap impact: confirms that this is a classification task with mixed numeric and categorical predictors.

## 2) Summary statistics (key numeric features)
| stat | dep_hour | month | day_of_week | DISTANCE | w_tmax | w_prcp |
|---|---|---|---|---|---|---|
| count | 150000.00 | 150000.00 | 150000.00 | 150000.00 | 2765.00 | 2556.00 |
| mean | 13.47 | 3.59 | 2.99 | 809.49 | 20.17 | 3.48 |
| std | 4.93 | 1.70 | 2.01 | 588.12 | 10.36 | 9.26 |
| min | 0.02 | 1.00 | 0.00 | 31.00 | -22.70 | 0.00 |
| 25% | 9.17 | 2.00 | 1.00 | 383.00 | 12.80 | 0.00 |
| 50% | 13.38 | 4.00 | 3.00 | 657.00 | 20.60 | 0.00 |
| 75% | 17.58 | 5.00 | 5.00 | 1050.00 | 28.90 | 2.00 |
| max | 23.98 | 6.00 | 6.00 | 5095.00 | 45.00 | 90.40 |

- `dep_hour`: fractional hour (0–23) parsed from BTS `CRS_DEP_TIME`; spread across the full day.
- `DISTANCE`: right-skewed (short hops dominate; long-haul flights are outliers handled by RobustScaler).
- Weather columns (`w_*`) have high missing rates — median imputation applied in the model pipeline.

## 3) Target distribution (class balance)
- Delay prevalence: **21.02%** delayed vs **78.98%** not delayed.
- Interpretation: class imbalance is present, so model evaluation should emphasize precision/recall/F1 (not accuracy alone).
- Roadmap impact: use imbalance-aware evaluation and keep probability outputs for threshold tuning.

## 4) Distribution and pattern checks
- Time-of-day effect: highest delay risk appears around hour **20:00** at **30.86%**.
- Carrier-level variation indicates airline identity contributes useful signal.
- Correlation matrix helps flag redundant numeric predictors before modeling.
- Roadmap impact: retain hour and carrier features; monitor multicollinearity among numeric variables.

## 5) Outlier check and handling rationale
- Method: IQR rule (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`) on key numeric variables.
- `w_prcp`: 463/2,556 potential outliers (18.11%).
- `DISTANCE`: 8,424/150,000 potential outliers (5.62%).
- `w_tmax`: 3/2,765 potential outliers (0.11%).
- `dep_hour`: 0/150,000 potential outliers (0.00%).
- Handling policy: keep valid operational extremes (e.g., long-distance flights, extreme weather) and rely on robust models/tree splits; add clipping only if validation metrics degrade.
- Roadmap impact: preserves real rare events while keeping a documented mitigation strategy.

## 6) EDA conclusion
- EDA supports the modeling approach: mixed-feature classification with interpretable risk outputs.
- Next steps: continue threshold tuning and error analysis by carrier, hour, and weather regime.
