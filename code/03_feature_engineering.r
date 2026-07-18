# ============================================================
# 03_feature_engineering.r
# Project : Financial Inclusion in Africa — Bank Account Prediction
# Purpose : Engineer new features from raw variables:
#           binning, categorization, interactions.
# Requires: df, dt  (02_data_cleaning.r)
# Outputs : df, dt  (with engineered features)
# ============================================================

cat("\n========== 03: FEATURE ENGINEERING ==========\n\n")

# ============================================================
# AGE BINNING
# ============================================================
cat("Creating age bins...\n")

df$age <- NA
df$age[df$age_of_respondent <= 20] <- "20"
df$age[df$age_of_respondent >= 21 & df$age_of_respondent <= 25] <- "25"
df$age[df$age_of_respondent >= 26 & df$age_of_respondent <= 30] <- "30"
df$age[df$age_of_respondent >= 31 & df$age_of_respondent <= 60] <- "60"
df$age[df$age_of_respondent >= 61 & df$age_of_respondent <= 80] <- "80"
df$age[df$age_of_respondent >= 81] <- "90"
df$age <- as.factor(df$age)

dt$age <- NA
dt$age[dt$age_of_respondent <= 20] <- "20"
dt$age[dt$age_of_respondent >= 21 & dt$age_of_respondent <= 25] <- "25"
dt$age[dt$age_of_respondent >= 26 & dt$age_of_respondent <= 30] <- "30"
dt$age[dt$age_of_respondent >= 31 & dt$age_of_respondent <= 60] <- "60"
dt$age[dt$age_of_respondent >= 61 & dt$age_of_respondent <= 80] <- "80"
dt$age[dt$age_of_respondent >= 81] <- "90"
dt$age <- as.factor(dt$age)

# ============================================================
# HOUSEHOLD SIZE CATEGORIZATION
# ============================================================
cat("Creating household size categories...\n")

df$house <- NA
df$house[df$household_size == 1] <- "S"
df$house[df$household_size >= 2 & df$household_size <= 5] <- "M"
df$house[df$household_size > 5] <- "L"
df$house <- as.factor(df$house)

dt$house <- NA
dt$house[dt$household_size == 1] <- "S"
dt$house[dt$household_size >= 2 & dt$household_size <= 5] <- "M"
dt$house[dt$household_size > 5] <- "L"
dt$house <- as.factor(dt$house)

# ============================================================
# INCOME SOURCE CATEGORIZATION
# ============================================================
cat("Creating income source categories...\n")

df$source <- NA
df$source[df$job_type %in% c("Formally employed Government",
                             "Formally employed Private")] <- "Formally"
df$source[df$job_type %in% c("Self employed", "Other Income",
                             "Dont Know/Refuse to answer")] <- "Business"
df$source[df$job_type %in% c("Farming and Fishing",
                             "Informally employed")] <- "Farming"
df$source[df$job_type %in% c("Government Dependent",
                             "Remittance Dependent")] <- "Dependent"
df$source[df$job_type == "No Income"] <- "Not"
df$source <- as.factor(df$source)

dt$source <- NA
dt$source[dt$job_type %in% c("Formally employed Government",
                             "Formally employed Private")] <- "Formally"
dt$source[dt$job_type %in% c("Self employed", "Other Income",
                             "Dont Know/Refuse to answer")] <- "Business"
dt$source[dt$job_type %in% c("Farming and Fishing",
                             "Informally employed")] <- "Farming"
dt$source[dt$job_type %in% c("Government Dependent",
                             "Remittance Dependent")] <- "Dependent"
dt$source[dt$job_type == "No Income"] <- "Not"
dt$source <- as.factor(dt$source)

# ============================================================
# GEOGRAPHIC INTERACTIONS
# ============================================================
cat("Creating geographic interactions...\n")

df$land <- paste0(df$country, ".", df$location_type)
df$land <- as.factor(df$land)

dt$land <- paste0(dt$country, ".", dt$location_type)
dt$land <- as.factor(dt$land)

# ============================================================
# DEMOGRAPHIC INTERACTIONS
# ============================================================
cat("Creating demographic interactions...\n")

df$human <- paste0(df$country, ".", df$gender_of_respondent)
df$human <- as.factor(df$human)

dt$human <- paste0(dt$country, ".", dt$gender_of_respondent)
dt$human <- as.factor(dt$human)

# ============================================================
# SAVINGS BEHAVIOR CATEGORIZATION
# ============================================================
cat("Creating savings behavior categories...\n")

df$save <- NA
df$save[df$job_type %in% c("Formally employed Government",
                           "Formally employed Private",
                           "Self employed")] <- "Bank"
df$save[df$job_type %in% c("Farming and Fishing",
                           "Other Income")] <- "Other"
df$save[df$job_type %in% c("No Income", "Informally employed",
                           "Dont Know/Refuse to answer")] <- "Hand"
df$save <- as.factor(df$save)

dt$save <- NA
dt$save[dt$job_type %in% c("Formally employed Government",
                           "Formally employed Private",
                           "Self employed")] <- "Bank"
dt$save[dt$job_type %in% c("Farming and Fishing",
                           "Other Income")] <- "Other"
dt$save[dt$job_type %in% c("No Income", "Informally employed",
                           "Dont Know/Refuse to answer")] <- "Hand"
dt$save <- as.factor(dt$save)

# ============================================================
# INCOME FREQUENCY CATEGORIZATION
# ============================================================
cat("Creating income frequency categories...\n")

df$income <- NA
df$income[df$job_type %in% c("Formally employed Government",
                             "Formally employed Private")] <- "Monthly"
df$income[df$job_type %in% c("Self employed")] <- "Daily"
df$income[df$job_type %in% c("Farming and Fishing")] <- "Seasonally"
df$income[df$job_type %in% c("Government Dependent",
                             "Remittance Dependent")] <- "Remittance"
df$income[df$job_type %in% c("No Income", "Other Income",
                             "Informally employed",
                             "Dont Know/Refuse to answer")] <- "Not"
df$income <- as.factor(df$income)

dt$income <- NA
dt$income[dt$job_type %in% c("Formally employed Government",
                             "Formally employed Private")] <- "Monthly"
dt$income[dt$job_type %in% c("Self employed")] <- "Daily"
dt$income[dt$job_type %in% c("Farming and Fishing")] <- "Seasonally"
dt$income[dt$job_type %in% c("Government Dependent",
                             "Remittance Dependent")] <- "Remittance"
dt$income[dt$job_type %in% c("No Income", "Other Income",
                             "Informally employed",
                             "Dont Know/Refuse to answer")] <- "Not"
dt$income <- as.factor(dt$income)

cat("\n✓ 03_feature_engineering.r complete\n")
cat("  Engineered features: age, house, source, land, human, save, income\n\n")
