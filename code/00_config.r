# ============================================================
# 00_config.r
# Project : Financial Inclusion in Africa — Bank Account Prediction
# Purpose : Load all libraries, set paths, seeds, constants,
#           and utility functions used throughout the pipeline.
# ============================================================

cat("\n========== 00: CONFIGURATION ==========\n\n")

# ============================================================
# DIRECTORIES & FILE PATHS
# ============================================================
DATA_DIR     <- "."              # Data files (Train_v2.csv, Test_v2.csv)
SUBMIT_DIR   <- "./submissions"  # Output directory for submission CSV
if (!dir.exists(SUBMIT_DIR)) dir.create(SUBMIT_DIR)

# ============================================================
# SEEDS & RANDOMIZATION
# ============================================================
PRIMARY_SEED <- 123
MODEL_SEED   <- 45
ENSEMBLE_SEED <- 1
set.seed(PRIMARY_SEED)

# ============================================================
# LIBRARIES — CORE DATA MANIPULATION
# ============================================================
library(here)
library(dplyr)
library(data.table)
library(tibble)
library(purrr)
library(stringr)

# ============================================================
# LIBRARIES — VISUALIZATION & EXPLORATION
# ============================================================
library(ggplot2)
library(rattle)

# ============================================================
# LIBRARIES — MACHINE LEARNING & MODELING
# ============================================================
library(caret)
library(mlr)
library(mlbench)
library(recipes)
library(randomForest)
library(ranger)
library(rpart)
library(party)
library(xgboost)
library(gbm)
library(e1071)
library(MASS)
library(nnet)
library(naivebayes)

# ============================================================
# LIBRARIES — FEATURE SELECTION & ENGINEERING
# ============================================================
library(Boruta)
library(RFE)

# ============================================================
# LIBRARIES — ENSEMBLE & STACKING
# ============================================================
library(caretEnsemble)
library(caretStack)

# ============================================================
# LIBRARIES — EVALUATION & IMPORTANCE
# ============================================================
library(InformationValue)
library(car)
library(GoodmanKruskal)

# ============================================================
# LIBRARIES — ENCODING & PREPROCESSING
# ============================================================
library(scorecard)

# ============================================================
# LIBRARIES — PARALLEL COMPUTING
# ============================================================
library(doParallel)
registerDoParallel(cores = 4)

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
TARGET_VAR           <- "bank_account"
TEST_ID_VAR          <- "uniqueid"
COUNTRIES            <- c("Kenya", "Rwanda", "Tanzania", "Uganda")
CV_FOLDS             <- 5
CV_REPEATS           <- 3
RANDOM_FOREST_NTREE  <- 500
XGBOOST_NROUNDS      <- 500
ENSEMBLE_METHOD      <- "average"  # "average", "stacked", "weighted"

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

# Calculate classification metrics
calc_metrics <- function(actual, predicted, name = "Model") {
    pred_class <- ifelse(predicted > 0.5, "Yes", "No")
    actual_class <- actual

    acc <- mean(pred_class == actual_class)
    auc <- tryCatch(
        InformationValue::AUROC(actual_class == "Yes", predicted),
        error = function(e) NA
    )

    cat(sprintf("  %s | Accuracy: %.4f | AUC: %.4f\n", name, acc, auc))
    list(accuracy = acc, auc = auc)
}

# Create stratified folds for CV
create_folds <- function(y, k = 5, seed = 123) {
    set.seed(seed)
    createFolds(y, k = k, list = TRUE, returnTrain = TRUE)
}

# Safe model training wrapper
safe_train <- function(expr, model_name) {
    tryCatch(expr,
        error = function(e) {
            cat("✗", model_name, ":", e$message, "\n")
            NULL
        }
    )
}

cat("✓ 00_config.r complete\n")
cat("  Libraries loaded, paths configured, seeds set\n\n")
