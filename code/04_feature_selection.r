# ============================================================
# 04_feature_selection.r
# Project : Financial Inclusion in Africa — Bank Account Prediction
# Purpose : Identify most predictive features using Boruta,
#           RFE, and importance-based methods.
# Requires: df, dt  (03_feature_engineering.r)
# Outputs : selected_features (final list), fi_summary (importance)
# ============================================================

cat("\n========== 04: FEATURE SELECTION ==========\n\n")

# ============================================================
# PREPARE DATA FOR FEATURE SELECTION
# ============================================================
cat("Preparing data for feature selection...\n")

# Remove target and ID columns
X <- df %>%
    select(-all_of(TARGET_VAR), -uniqueid) %>%
    mutate(across(where(is.character), as.factor))

y <- factor(df[[TARGET_VAR]])

cat("  Features available:", ncol(X), "\n")
cat("  Samples:", nrow(X), "\n\n")

# ============================================================
# BORUTA FEATURE SELECTION
# ============================================================
cat("Running Boruta feature selection (100 iterations)...\n")

boruta_result <- tryCatch({
    Boruta(X, y, doTrace = 0, maxRuns = 100)
}, error = function(e) {
    cat("  ✗ Boruta failed:", e$message, "\n")
    NULL
})

if (!is.null(boruta_result)) {
    boruta_confirmed <- names(boruta_result$finalDecision)[
        boruta_result$finalDecision == "Confirmed"
    ]
    cat("  Confirmed features:", length(boruta_confirmed), "\n")
    cat("    ", paste(head(boruta_confirmed, 10), collapse = ", "), "\n\n")
}

# ============================================================
# RANDOM FOREST IMPORTANCE
# ============================================================
cat("Computing Random Forest feature importance...\n")

rf_importance <- tryCatch({
    set.seed(MODEL_SEED)
    rf_model <- randomForest(X, y, ntree = 200, importance = TRUE)
    importance(rf_model, type = 2) %>%
        as.data.frame() %>%
        rownames_to_column("feature") %>%
        arrange(desc(MeanDecreaseGini))
}, error = function(e) {
    cat("  ✗ RF importance failed:", e$message, "\n")
    NULL
})

if (!is.null(rf_importance)) {
    cat("  Top 15 features (RF importance):\n")
    print(head(rf_importance, 15))
    cat("\n")
}

# ============================================================
# CORRELATION WITH TARGET
# ============================================================
cat("Computing feature-target correlations...\n")

# Convert factors to numeric
X_numeric <- X %>%
    mutate(across(where(is.factor), as.numeric))
y_numeric <- as.numeric(y) - 1

correlations <- sapply(X_numeric, function(x) {
    abs(cor(x, y_numeric, use = "complete.obs"))
})

cor_df <- data.frame(
    feature = names(correlations),
    correlation = correlations,
    stringsAsFactors = FALSE
) %>%
    arrange(desc(correlation))

cat("  Top 15 features (correlation with target):\n")
print(head(cor_df, 15))
cat("\n")

# ============================================================
# COMBINE SELECTIONS
# ============================================================
cat("Combining feature selections...\n")

# Top features from each method
rf_top <- head(rf_importance$feature, 20)
cor_top <- head(cor_df$feature, 20)
boruta_top <- if (!is.null(boruta_result)) head(boruta_confirmed, 20) else character(0)

selected_features <- unique(c(rf_top, cor_top, boruta_top))
selected_features <- selected_features[!is.na(selected_features)]

cat("  Final feature set:", length(selected_features), "features\n\n")

# ============================================================
# SAVE FEATURE IMPORTANCE SUMMARY
# ============================================================
fi_summary <- list(
    rf_importance = rf_importance,
    correlation = cor_df,
    boruta_result = boruta_result,
    selected_features = selected_features
)

cat("✓ 04_feature_selection.r complete\n")
cat("  Objects: selected_features, fi_summary\n\n")
