# Financial Inclusion in Africa — Bank Account Prediction Pipeline

This competition is hosted on Zindi, a machine learning platform for data science challenges.  
Here is the link to the competition: [Financial Inclusion in Africa 🌾 - Knowledge](https://zindi.africa/competitions/financial-inclusion-in-africa)

Ranked in the TOP 7%
---

Organized classification pipeline · Binary prediction · Multiple ML models with ensemble.

---

## Competition Overview

| Item | Details |
|---|---|
| Task | Predict whether an individual has or uses a **bank account** |
| Target | Binary classification: Yes / No |
| Geography | Multiple African countries (Kenya, Rwanda, Tanzania, Uganda) |
| Features | Demographics, socioeconomic, household, access indicators |
| Evaluation | Likely accuracy or ROC-AUC (standard binary classification) |
| Submission format | One row per `uniqueid` with predicted probabilities |

---

## Dataset

| File | Rows | Columns | Description |
|---|---|---|---|
| `Train_v2.csv` | ~33,000 | 10+ | Training set with target `bank_account` |
| `Test_v2.csv` | ~14,000 | 9+ | Test set — `bank_account` withheld |

**Key Features:**
- **Geography:** `country`, `location_type` (Rural/Urban)
- **Demographics:** `age_of_respondent`, `gender_of_respondent`, `marital_status`, `relationship_with_head`
- **Socioeconomic:** `job_type`, `education_level`, `household_size`
- **Access:** `cellphone_access`
- **Target:** `bank_account` (Yes/No)

---

## Pipeline Structure

```
pipeline/
├── 00_config.r                # Libraries, paths, constants, seeds, utilities
├── 01_data_loading.r          # Load Train_v2.csv and Test_v2.csv
├── 02_data_cleaning.r         # Remove inconsistencies, impute, consolidate
├── 03_feature_engineering.r   # Binning, interactions (age, income, geo, demo)
├── 04_feature_selection.r     # Boruta, RFE, correlation analysis
├── 05_preprocessing.r         # Encoding, scaling, prepare for models
├── 06_models.r                # Train RF, Ranger, XGBoost, GLM, SVM, NB, GBM
├── 07_evaluation.r            # Model leaderboard, holdout set validation
├── 08_submission.r            # Ensemble, save final submission CSV
├── MAIN.r                      # Orchestration — run this to execute full pipeline
└── README.md                   # This file
```

---

## Quick Start

```r
# Run the full pipeline
setwd("path/to/pipeline")
source("MAIN.r")
```

The submission CSV is written to `./submissions/submission_YYYY-MM-DD.csv`.

---

## Pipeline Stages

### **00 — Configuration** (`00_config.r`)
- Load all required libraries (dplyr, caret, xgboost, ranger, e1071, etc.)
- Set paths, seeds (`PRIMARY_SEED=123`, `MODEL_SEED=45`)
- Define constants: `TARGET_VAR="bank_account"`, `COUNTRIES`, `CV_FOLDS=5`
- Register parallel backend (4 cores)

### **01 — Data Loading** (`01_data_loading.r`)
- Load `Train_v2.csv` → `df`
- Load `Test_v2.csv` → `dt`
- Backup test data → `dtu` (preserve test IDs)

### **02 — Data Cleaning** (`02_data_cleaning.r`)
- Remove `year` and `uniqueid` columns
- Remove inconsistent records (bank_account=Yes but job_type=No Income)
- Impute missing `marital_status` based on `household_size` and `relationship_with_head`
- Consolidate categories (e.g., Other non-relatives → Other relative)

### **03 — Feature Engineering** (`03_feature_engineering.r`)
- **Age binning:** 20, 25, 30, 60, 80, 90 (categorical)
- **Household categorization:** S (≤1), M (2-5), L (>5)
- **Income source:** Formally, Business, Farming, Dependent, Not
- **Geographic interactions:** country × location_type
- **Demographic interactions:** country × gender
- **Savings behavior:** Bank, Other, Hand
- **Income frequency:** Monthly, Daily, Seasonally, Remittance, Not

### **04 — Feature Selection** (`04_feature_selection.r`)
- **Boruta:** Identify confirmed features (100 iterations)
- **Random Forest importance:** Mean Decrease Gini ranking
- **Correlation analysis:** Feature-target correlations
- **Combine:** Merge selections into final feature set

### **05 — Preprocessing** (`05_preprocessing.r`)
- Select model-ready features
- Handle missing values (median for numeric, mode for categorical)
- **One-hot encoding:** For tree models
- **Scaling/normalization:** For linear models (GLM, SVM)
- Ensure factor levels match train/test

### **06 — Models** (`06_models.r`)
Train 7 diverse classifiers:

| Model | Library | Key Settings |
|---|---|---|
| Random Forest | `randomForest` | ntree=500, importance=TRUE |
| Ranger | `ranger` | num.trees=500, probability=TRUE |
| XGBoost | `xgboost` | max_depth=6, eta=0.1, nrounds=500 |
| Logistic Regression | `glm` | family="binomial" |
| SVM | `e1071` | kernel="radial", cost=5 |
| Naive Bayes | `naivebayes` | Laplace=1 |
| GBM | `gbm` | distribution="bernoulli", n.trees=500 |

### **07 — Evaluation** (`07_evaluation.r`)
- Create holdout set (20% of training) for validation
- Re-train each model on holdout train
- Evaluate on holdout test
- **Leaderboard:** Rank models by accuracy and AUC

### **08 — Submission** (`08_submission.r`)
- **Ensemble method:** Average predictions from all trained models (default)
- Clip probabilities to [0, 1]
- Create submission with columns:
  - `ID`, `TX_07_AUC`, `TX_07_LogLoss`, `TX_90_AUC`, `TX_90_LogLoss`, `TX_120_AUC`, `TX_120_LogLoss`
- Save to `./submissions/submission_YYYY-MM-DD.csv`

---

## Key Design Decisions

### Data Processing
- **Target encoding:** "Yes" → 1, "No" → 0
- **Missing values:** Numeric → median, Categorical → mode
- **Factor levels:** Train levels applied to test set
- **Class imbalance:** Monitored but not explicitly addressed (can add SMOTE if needed)

### Modeling
- **Multiple algorithms:** Tree-based (RF, Ranger, XGBoost), Linear (GLM), Kernel (SVM), Bayesian (NB), Ensemble (GBM)
- **Ensemble:** Simple average of all model predictions (robust, reduces variance)
- **Parallel processing:** 4 cores registered for speed

### Validation
- **Holdout validation:** 20% held out, 80% training
- **Internal CV:** Models can use caret's cross-validation for tuning (not fully implemented here)

---

## Requirements

```r
install.packages(c(
    # Core
    "here", "dplyr", "data.table", "tibble", "purrr", "stringr",
    # Visualization
    "ggplot2", "rattle",
    # ML & Modeling
    "caret", "mlr", "mlbench", "recipes",
    "randomForest", "ranger", "rpart", "party",
    "xgboost", "gbm", "e1071", "MASS", "nnet", "naivebayes",
    # Feature Selection
    "Boruta",
    # Ensemble
    "caretEnsemble", "caretStack",
    # Evaluation
    "InformationValue", "car", "GoodmanKruskal",
    # Encoding
    "scorecard",
    # Parallel
    "doParallel"
))
```

---

## Submission Format

Expected output CSV:
```
ID,TX_07_AUC,TX_07_LogLoss,TX_90_AUC,TX_90_LogLoss,TX_120_AUC,TX_120_LogLoss
ID_KYOSSX,0.75,0.75,0.68,0.68,0.62,0.62
ID_AXYZBC,0.42,0.42,0.51,0.51,0.55,0.55
```

Both AUC and LogLoss columns contain the same predicted probability (competition evaluates each separately).

---

## Running Individual Stages

To run stages interactively instead of the full pipeline:

```r
# Load configuration first
source("00_config.r")

# Then run any combination:
source("01_data_loading.r")
source("02_data_cleaning.r")
source("03_feature_engineering.r")
source("04_feature_selection.r")
source("05_preprocessing.r")
source("06_models.r")
source("07_evaluation.r")
source("08_submission.r")
```

---

## Notes

- **Data paths:** Ensure `Train_v2.csv` and `Test_v2.csv` are in the working directory
- **Reproducibility:** Seeds are set (123, 45, 1) but parallel processing may introduce minor variations
- **Computation time:** ~5-15 minutes on standard hardware (depends on CPU cores)
- **Memory:** Keep models in memory during pipeline; can be cleared per stage if needed
