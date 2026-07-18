# ============================================================
# 08_submission.r
# Project : Financial Inclusion in Africa — Bank Account Prediction
# Purpose : Build ensemble from model predictions,
#           generate final submission CSV.
# Requires: model_predictions, leaderboard_df
# Outputs : submission CSV in SUBMIT_DIR
# ============================================================

cat("\n========== 08: SUBMISSION ==========\n\n")

# ============================================================
# SELECT ENSEMBLE METHOD
# ============================================================
cat("Building ensemble predictions...\n")
cat("  Method:", ENSEMBLE_METHOD, "\n\n")

# Get prediction columns (exclude uniqueid and country)
pred_cols <- setdiff(names(model_predictions), c("uniqueid", "country"))

if (length(pred_cols) == 0) {
    cat("✗ No model predictions found!\n\n")
} else {
    cat("  Available models:", length(pred_cols), "\n")
    cat("    ", paste(pred_cols, collapse = ", "), "\n\n")

    # ========================================================
    # METHOD 1: AVERAGE ENSEMBLE
    # ========================================================
    if (ENSEMBLE_METHOD == "average") {
        cat("Computing average ensemble...\n")

        pred_matrix <- as.matrix(model_predictions[, pred_cols])
        ensemble_pred <- rowMeans(pred_matrix, na.rm = TRUE)
    }

    # ========================================================
    # METHOD 2: WEIGHTED ENSEMBLE (by leaderboard rank)
    # ========================================================
    else if (ENSEMBLE_METHOD == "weighted") {
        cat("Computing weighted ensemble (by AUC)...\n")

        if (exists("leaderboard_df") && nrow(leaderboard_df) > 0) {
            # Create weight map from leaderboard
            leaderboard_df$weight <- (nrow(leaderboard_df):1) / sum(1:nrow(leaderboard_df))
            weight_map <- setNames(leaderboard_df$weight,
                                  tolower(gsub(" ", "_", leaderboard_df$model)))

            pred_matrix <- as.matrix(model_predictions[, pred_cols])
            weights <- sapply(pred_cols, function(col) {
                weight_map[col] %||% (1 / length(pred_cols))
            })
            weights <- weights / sum(weights)

            ensemble_pred <- rowSums(pred_matrix * matrix(weights,
                                                           nrow = nrow(pred_matrix),
                                                           ncol = ncol(pred_matrix),
                                                           byrow = TRUE))
        } else {
            cat("  ⚠ Leaderboard not available, falling back to averaging\n")
            pred_matrix <- as.matrix(model_predictions[, pred_cols])
            ensemble_pred <- rowMeans(pred_matrix, na.rm = TRUE)
        }
    }

    # ========================================================
    # DEFAULT: AVERAGE
    # ========================================================
    else {
        cat("Computing average ensemble (default)...\n")
        pred_matrix <- as.matrix(model_predictions[, pred_cols])
        ensemble_pred <- rowMeans(pred_matrix, na.rm = TRUE)
    }

    # ========================================================
    # CLIP PREDICTIONS TO [0, 1]
    # ========================================================
    ensemble_pred <- pmax(pmin(ensemble_pred, 1.0), 0.0)

    cat("  Ensemble predictions: min =", round(min(ensemble_pred), 4),
        "| max =", round(max(ensemble_pred), 4),
        "| mean =", round(mean(ensemble_pred), 4), "\n\n")

    # ========================================================
    # BUILD SUBMISSION DATAFRAME
    # ========================================================
    cat("Building submission file...\n")

    # For this competition, both AUC and LogLoss columns get the same probability
    submission <- data.frame(
        ID = dtu[[TEST_ID_VAR]],
        TX_07_AUC = round(ensemble_pred, 6),
        TX_07_LogLoss = round(ensemble_pred, 6),
        TX_90_AUC = round(ensemble_pred, 6),
        TX_90_LogLoss = round(ensemble_pred, 6),
        TX_120_AUC = round(ensemble_pred, 6),
        TX_120_LogLoss = round(ensemble_pred, 6),
        stringsAsFactors = FALSE
    )

    cat("  Submission rows:", nrow(submission), "\n")
    cat("  Submission columns:", ncol(submission), "\n")
    cat("  Sample predictions:\n")
    print(head(submission, 10))
    cat("\n")

    # ========================================================
    # SAVE SUBMISSION
    # ========================================================
    submit_file <- file.path(SUBMIT_DIR,
        paste0("submission_", format(Sys.Date(), "%Y-%m-%d"), ".csv"))

    write.csv(submission, submit_file, row.names = FALSE)

    cat("✓ Submission saved to:", submit_file, "\n\n")
}

cat("✓ 08_submission.r complete\n\n")
