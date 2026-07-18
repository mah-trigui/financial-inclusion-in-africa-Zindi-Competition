# ============================================================
# MAIN.r
# Project : Financial Inclusion in Africa — Bank Account Prediction
# Purpose : Full end-to-end pipeline orchestrator.
#           Run this file to reproduce all results from
#           raw data to final submission CSV.
# ============================================================

cat("╔════════════════════════════════════════════════════════╗\n")
cat("║  FINANCIAL INCLUSION — BANK ACCOUNT PREDICTION        ║\n")
cat("║  Complete ML Pipeline                                 ║\n")
cat("╚════════════════════════════════════════════════════════╝\n\n")

# Locate the pipeline directory
PIPELINE_DIR <- dirname(sys.frame(1)$ofile)
if (!nchar(PIPELINE_DIR)) PIPELINE_DIR <- getwd()  # fallback (interactive)

source_step <- function(file, step_label) {
    cat(strrep("═", 58), "\n")
    cat("  STEP:", step_label, "\n")
    cat(strrep("═", 58), "\n")
    t0 <- proc.time()
    source(file.path(PIPELINE_DIR, file), echo = FALSE)
    elapsed <- (proc.time() - t0)["elapsed"]
    cat("  ↳ Completed in", round(elapsed, 1), "seconds\n\n")
}

# ── 00 Configuration ─────────────────────────────────────
source_step("00_config.r",            "00 · Configuration & libraries")

# ── 01 Data Loading ──────────────────────────────────────
source_step("01_data_loading.r",      "01 · Load raw data")

# ── 02 Data Cleaning ─────────────────────────────────────
source_step("02_data_cleaning.r",     "02 · Data cleaning & consolidation")

# ── 03 Feature Engineering ───────────────────────────────
source_step("03_feature_engineering.r", "03 · Feature engineering")

# ── 04 Feature Selection ─────────────────────────────────
source_step("04_feature_selection.r", "04 · Feature selection (Boruta/RFE)")

# ── 05 Preprocessing ─────────────────────────────────────
source_step("05_preprocessing.r",     "05 · Encoding & scaling")

# ── 06 Models ────────────────────────────────────────────
source_step("06_models.r",            "06 · Train models")

# ── 07 Evaluation ────────────────────────────────────────
source_step("07_evaluation.r",        "07 · Evaluate & rank models")

# ── 08 Submission ────────────────────────────────────────
source_step("08_submission.r",        "08 · Generate submission")

# ── Done ─────────────────────────────────────────────────
cat("╔════════════════════════════════════════════════════════╗\n")
cat("║                  PIPELINE COMPLETE                    ║\n")
cat("╚════════════════════════════════════════════════════════╝\n\n")
cat("Submission saved to:", SUBMIT_DIR, "\n\n")
