# System Architecture — Fertility Outcome Classifier

## Overview
End-to-end clinical ML system predicting pregnancy outcome (Success/Failure) from couples' health, lifestyle, and treatment features. Deployed as a containerized REST API on Azure Container Apps.

## Data Flow

```
Raw CSV (14 cols, 800 rows)
    -> Drop Couple_ID (no signal)
    -> Fill Treatment_Type NaN -> "None" (no treatment received)
    -> Fill Alcohol_Intake NaN -> "None" (no alcohol reported)
    -> Engineer Female_Age_x_Motility interaction
    -> 13 features total
        |-- 6 categorical  -> OneHotEncoder(drop=first)
        |-- 7 numeric      -> StandardScaler
    -> XGBoost classifier (scale_pos_weight=0.37)
    -> Pregnancy_Outcome (Success=1 / Failure=0) + probability
```

## Informative NaN Strategy

| Column | NaN Count | Fill | Clinical Meaning |
|---|---|---|---|
| Treatment_Type | 500 (62.5%) | "None" | No fertility treatment received |
| Alcohol_Intake | 259 (32.4%) | "None" | No alcohol consumption reported |

These are NOT imputed statistically. The missingness itself carries clinical information.

## Engineered Feature

**Female_Age_x_Motility** = Female_Age × Motility_%

Captures age-adjusted reproductive potential. As female age increases, the compounding effect of sperm motility on outcomes becomes more pronounced. This interaction is clinically motivated by reproductive endocrinology literature.

## Decision Threshold

Default 0.5 -> Tuned to **0.40** for Failure class recall.

In fertility counseling, classifying a high-risk couple as "likely Success" delays intervention and reduces the treatment window. Lower threshold increases sensitivity for the Failure class — the clinical priority.
