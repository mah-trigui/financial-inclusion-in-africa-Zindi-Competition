# ============================================================
# 07_evaluation.r
# Project : Financial Inclusion in Africa — Bank Account Prediction
# Purpose : Evaluate individual models, create leaderboard,
#           compare performance across test set.
# Requires: model_predictions, df_model
# Outputs : leaderboard_df (model rankings)
# ============================================================

cat("\n========== 07: EVALUATION ==========\n\n")

# ============================================================
# PREPARE HOLDOUT DATA FOR EVALUATION
# ============================================================
cat("Preparing evaluation dataset...\n")

# Use a holdout set from training data for internal validation
set.seed(PRIMARY_SEED)
holdout_idx <- createDataPartition(
    df_model[[TARGET_VAR]],
    p = 0.2,
    list = FALSE
)

df_eval_train <- df_model[-holdout_idx, ]
df_eval_test <- df_model[holdout_idx, ]

y_eval_test <- as.numeric(df_eval_test[[TARGET_VAR]] == "Yes")

cat("  Holdout train:", nrow(df_eval_train), "rows\n")
cat("  Holdout test:", nrow(df_eval_test), "rows\n\n")

# ============================================================
# RE-EVALUATE MODELS ON HOLDOUT SET
# ============================================================
cat("Evaluating models on holdout set...\n\n")

leaderboard_rows <- list()

# ────────── Random Forest ──────────
tryCatch({
    set.seed(MODEL_SEED)
    rf_eval <- randomForest(
        x = df_eval_train %>% select(-all_of(TARGET_VAR)),
        y = factor(df_eval_train[[TARGET_VAR]]),
        ntree = RANDOM_FOREST_NTREE
    )
    rf_eval_pred <- predict(rf_eval, df_eval_test, type = "prob")[, "Yes"]
    metrics_rf <- calc_metrics(df_eval_test[[TARGET_VAR]], rf_eval_pred, "Random Forest")
    leaderboard_rows$rf <- data.frame(
        model = "Random Forest",
        accuracy = metrics_rf$accuracy,
        auc = metrics_rf$auc
    )
}, error = function(e) cat("  ✗ RF eval failed\n"))

# ────────── Ranger ──────────
tryCatch({
    set.seed(MODEL_SEED)
    ranger_eval <- ranger(
        formula = as.formula(paste(TARGET_VAR, "~ .")),
        data = df_eval_train,
        num.trees = RANDOM_FOREST_NTREE,
        probability = TRUE
    )
    ranger_eval_pred <- predict(ranger_eval, df_eval_test)$predictions[, "Yes"]
    metrics_ranger <- calc_metrics(df_eval_test[[TARGET_VAR]], ranger_eval_pred, "Ranger")
    leaderboard_rows$ranger <- data.frame(
        model = "Ranger",
        accuracy = metrics_ranger$accuracy,
        auc = metrics_ranger$auc
    )
}, error = function(e) cat("  ✗ Ranger eval failed\n"))

# ────────── XGBoost ──────────
tryCatch({
    set.seed(MODEL_SEED)

    X_eval_train <- as.matrix(df_eval_train %>%
        select(-all_of(TARGET_VAR)) %>%
        mutate(across(where(is.factor), as.numeric)))
    X_eval_test <- as.matrix(df_eval_test %>%
        select(-all_of(TARGET_VAR)) %>%
        mutate(across(where(is.factor), as.numeric)))

    xgb_eval <- xgboost(
        data = X_eval_train,
        label = as.numeric(df_eval_train[[TARGET_VAR]] == "Yes"),
        nrounds = 300,
        max_depth = 6,
        eta = 0.1,
        objective = "binary:logistic",
        verbose = 0
    )
    xgb_eval_pred <- predict(xgb_eval, X_eval_test)
    metrics_xgb <- calc_metrics(df_eval_test[[TARGET_VAR]], xgb_eval_pred, "XGBoost")
    leaderboard_rows$xgboost <- data.frame(
        model = "XGBoost",
        accuracy = metrics_xgb$accuracy,
        auc = metrics_xgb$auc
    )
}, error = function(e) cat("  ✗ XGBoost eval failed\n"))

# ────────── Logistic Regression ──────────
tryCatch({
    set.seed(MODEL_SEED)
    glm_eval <- glm(
        formula = as.formula(paste(TARGET_VAR, "~ .")),
        data = df_eval_train,
        family = binomial(link = "logit")
    )
    glm_eval_pred <- predict(glm_eval, df_eval_test, type = "response")
    metrics_glm <- calc_metrics(df_eval_test[[TARGET_VAR]], glm_eval_pred, "GLM")
    leaderboard_rows$glm <- data.frame(
        model = "GLM",
        accuracy = metrics_glm$accuracy,
        auc = metrics_glm$auc
    )
}, error = function(e) cat("  ✗ GLM eval failed\n"))

# ──────────  SVM ──────────
tryCatch({
    set.seed(MODEL_SEED)
    svm_eval <- svm(
        formula = as.formula(paste(TARGET_VAR, "~ .")),
        data = df_eval_train,
        kernel = "radial",
        cost = 5,
        probability = TRUE
    )
    svm_eval_pred <- attr(
        predict(svm_eval, df_eval_test, probability = TRUE),
        "probabilities"
    )[, "Yes"]
    metrics_svm <- calc_metrics(df_eval_test[[TARGET_VAR]], svm_eval_pred, "SVM")
    leaderboard_rows$svm <- data.frame(
        model = "SVM",
        accuracy = metrics_svm$accuracy,
        auc = metrics_svm$auc
    )
}, error = function(e) cat("  ✗ SVM eval failed\n"))

# ──────────  Naive Bayes ──────────
tryCatch({
    set.seed(MODEL_SEED)
    nb_eval <- naivebayes::naive_bayes(
        formula = as.formula(paste(TARGET_VAR, "~ .")),
        data = df_eval_train
    )
    nb_eval_pred <- predict(nb_eval, df_eval_test, type = "prob")[, "Yes"]
    metrics_nb <- calc_metrics(df_eval_test[[TARGET_VAR]], nb_eval_pred, "Naive Bayes")
    leaderboard_rows$nb <- data.frame(
        model = "Naive Bayes",
        accuracy = metrics_nb$accuracy,
        auc = metrics_nb$auc
    )
}, error = function(e) cat("  ✗ NB eval failed\n"))

cat("\n")

# ============================================================
# LEADERBOARD
# ============================================================
leaderboard_df <- do.call(rbind, leaderboard_rows)
rownames(leaderboard_df) <- NULL

if (!is.null(leaderboard_df) && nrow(leaderboard_df) > 0) {
    leaderboard_df <- leaderboard_df[order(-leaderboard_df$auc), ]

    cat("═" %+% strrep("═", 50) %+% "\n")
    cat("  HOLDOUT LEADERBOARD (top models by AUC)\n")
    cat("═" %+% strrep("═", 50) %+% "\n")
    print(leaderboard_df)
    cat("\n")
}

cat("✓ 07_evaluation.r complete\n")
cat("  Object: leaderboard_df\n\n")
