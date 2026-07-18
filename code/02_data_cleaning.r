# ============================================================
# 02_data_cleaning.r
# Project : Financial Inclusion in Africa — Bank Account Prediction
# Purpose : Clean data, handle missing values, remove
#           inconsistencies, consolidate categories.
# Requires: df, dt  (01_data_loading.r)
# Outputs : df, dt  (cleaned)
# ============================================================

cat("\n========== 02: DATA CLEANING ==========\n\n")

# ============================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================
cat("Removing unnecessary columns...\n")
df$year <- NULL
dt$year <- NULL
df$uniqueid <- NULL

# ============================================================
# REMOVE INCONSISTENT RECORDS
# ============================================================
cat("Removing inconsistent records...\n")
rows_before <- nrow(df)
df <- df[!(df$bank_account == "Yes" & df$job_type == "No Income"), ]
rows_after <- nrow(df)
cat("  Removed", rows_before - rows_after,
    "records (bank='Yes' but job_type='No Income')\n")

# ============================================================
# IMPUTE MISSING MARITAL STATUS
# ============================================================
cat("Imputing missing marital_status...\n")

# Single-person households
df$marital_status[df$marital_status == "Dont know" &
                  df$household_size == 1] <- "Single/Never Married"
dt$marital_status[dt$marital_status == "Dont know" &
                  dt$household_size == 1] <- "Single/Never Married"

# Two-person households
df$marital_status[df$marital_status == "Dont know" &
                  df$household_size == 2] <- "Married/Living together"
dt$marital_status[dt$marital_status == "Dont know" &
                  dt$household_size == 2] <- "Married/Living together"

# 3+ person households — head of household
df$marital_status[df$marital_status == "Dont know" &
                  df$household_size >= 3 &
                  df$relationship_with_head == "Head of Household"] <- "Single/Never Married"
dt$marital_status[dt$marital_status == "Dont know" &
                  dt$household_size >= 3 &
                  dt$relationship_with_head == "Head of Household"] <- "Single/Never Married"

# 3+ person households — non-head
df$marital_status[df$marital_status == "Dont know" &
                  df$household_size >= 3 &
                  df$relationship_with_head != "Head of Household"] <- "Married/Living together"
dt$marital_status[dt$marital_status == "Dont know" &
                  dt$household_size >= 3 &
                  dt$relationship_with_head != "Head of Household"] <- "Married/Living together"

# ============================================================
# CONSOLIDATE SIMILAR CATEGORIES
# ============================================================
cat("Consolidating similar categories...\n")
df$relationship_with_head[df$relationship_with_head == "Other non-relatives"] <- "Other relative"
dt$relationship_with_head[dt$relationship_with_head == "Other non-relatives"] <- "Other relative"

# ============================================================
# CHECK FOR MISSING VALUES
# ============================================================
cat("\nMissing values in training data:\n")
missing_df <- sapply(df, function(x) sum(is.na(x)))
print(missing_df[missing_df > 0])

cat("\nMissing values in test data:\n")
missing_dt <- sapply(dt, function(x) sum(is.na(x)))
print(missing_dt[missing_dt > 0])

cat("\n✓ 02_data_cleaning.r complete\n")
cat("  Training:", nrow(df), "rows ×", ncol(df), "columns\n")
cat("  Test:", nrow(dt), "rows ×", ncol(dt), "columns\n\n")
