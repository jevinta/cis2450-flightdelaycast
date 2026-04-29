# EDA Findings Summary

## 1) Data context and project objective
- Objective: predict whether arrival delay is greater than 15 minutes (`is_delayed`).
- Dataset size: **150,000** rows.
- Key feature groups: schedule (`dep_hour`, `month`, `day_of_week`), route (`ORIGIN`, `DEST`, `DISTANCE`), carrier (`OP_CARRIER`), optional weather (`w_*`, `dw_*`).
- Roadmap impact: confirms that this is a classification task with mixed numeric and categorical predictors.

## 2) Target distribution (class balance)
- Delay prevalence: **23.34%** delayed vs **76.66%** not delayed.
- Interpretation: class imbalance is present, so model evaluation should emphasize precision/recall/F1 (not accuracy alone).
- Roadmap impact: use imbalance-aware evaluation and keep probability outputs for threshold tuning.

## 3) Distribution and pattern checks
- Time-of-day effect: highest delay risk appears around hour **18:00** at **29.02%**.
- Carrier-level variation indicates airline identity contributes useful signal.
- Correlation matrix helps flag redundant numeric predictors before modeling.
- Roadmap impact: retain hour and carrier features; monitor multicollinearity among numeric variables.

## 4) Outlier check and handling rationale
- Method: IQR rule (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`) on key numeric variables.
- `DISTANCE`: 8,054/150,000 potential outliers (5.37%).
- `dep_hour`: 0/150,000 potential outliers (0.00%).
- Handling policy: keep valid operational extremes (e.g., long-distance flights, extreme weather) and rely on robust models/tree splits; add clipping only if validation metrics degrade.
- Roadmap impact: preserves real rare events while keeping a documented mitigation strategy.

## 5) EDA conclusion
- EDA supports the modeling approach: mixed-feature classification with interpretable risk outputs.
- Next steps: continue threshold tuning and error analysis by carrier, hour, and weather regime.
