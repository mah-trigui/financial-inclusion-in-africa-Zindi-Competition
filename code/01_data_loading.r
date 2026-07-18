# ============================================================
# 01_data_loading.r
# Project : Financial Inclusion in Africa — Bank Account Prediction
# Purpose : Load training and test data from CSV files,
#           store test IDs for final submission.
# Requires: 00_config.r
# Outputs : df (training data), dt (test data), dtu (test backup)
# ============================================================

cat("\n========== 01: DATA LOADING ==========\n\n")

# ============================================================
# LOAD TRAINING DATA
# ============================================================
cat("Loading training data...\n")
df <- as.data.frame(read.csv(file.path(DATA_DIR, "Train_v2.csv")))
cat("  ", nrow(df), "rows ×", ncol(df), "columns\n")

# ============================================================
# LOAD TEST DATA
# ============================================================
cat("Loading test data...\n")
dt <- as.data.frame(read.csv(file.path(DATA_DIR, "Test_v2.csv")))
cat("  ", nrow(dt), "rows ×", ncol(dt), "columns\n")

# ============================================================
# BACKUP TEST DATA (with IDs for submission)
# ============================================================
dtu <- dt
cat("  Backup created (dtu) with test IDs preserved\n\n")

# ============================================================
# DATA STRUCTURE OVERVIEW
# ============================================================
cat("Training data structure:\n")
str(df, max.level = 1)

cat("\nTest data structure:\n")
str(dt, max.level = 1)

# ============================================================
# CHECK FOR BASIC ISSUES
# ============================================================
cat("\nTraining data summary:\n")
cat("  Rows:", nrow(df), "\n")
cat("  Columns:", ncol(df), "\n")
cat("  Target variable:", TARGET_VAR, "\n")
cat("  Target distribution:\n")
print(table(df[[TARGET_VAR]], useNA = "ifany"))

cat("\n✓ 01_data_loading.r complete\n")
cat("  Objects: df (training), dt (test), dtu (test backup)\n\n")
