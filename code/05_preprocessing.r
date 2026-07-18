# ============================================================
# 05_preprocessing.r
# Project : Financial Inclusion in Africa — Bank Account Prediction
# Purpose : Encode categorical variables, normalize/scale,
#           prepare data for modeling.
# Requires: df, dt, selected_features  (04_feature_selection.r)
# Outputs : df_model, dt_model (ready for training)
# ============================================================

cat("\n========== 05: PREPROCESSING ==========\n\n")

# ============================================================
# SELECT FEATURES FOR MODELING
# ============================================================
cat("Selecting features for modeling...\n")

# Use selected features + add some important engineered features
model_features <- c(
    "country", "location_type", "cellphone_access",
    "age_of_respondent", "gender_of_respondent", "relationship_with_head",
    "marital_status", "education_level", "job_type", "household_size",
    "age", "house", "source", "land", "human", "save", "income"
)

# Intersect with selected features
model_features <- intersect(model_features, c(selected_features, names(df)))
model_features <- unique(model_features)

cat("  ", length(model_features), "features selected\n\n")

# ============================================================
# CREATE MODELING DATASETS
# ============================================================
cat("Creating modeling datasets...\n")

df_model <- df %>%
    select(all_of(TARGET_VAR), all_of(model_features)) %>%
    mutate(across(where(is.character) & !matches(TARGET_VAR), as.factor))

dt_model <- dt %>%
    select(all_of(model_features)) %>%
    mutate(across(where(is.character), as.factor))

# Ensure factor levels match between train and test
for (col in intersect(names(df_model), names(dt_model))) {
    if (is.factor(df_model[[col]])) {
        levels(dt_model[[col]]) <- levels(df_model[[col]])
    }
}

cat("  Training features:", ncol(df_model) - 1, "| Rows:", nrow(df_model), "\n")
cat("  Test features:", ncol(dt_model), "| Rows:", nrow(dt_model), "\n\n")

# ============================================================
# HANDLE MISSING VALUES (if any)
# ============================================================
cat("Handling missing values...\n")

# Fill missing numeric with median
for (col in names(df_model)) {
    if (is.numeric(df_model[[col]])) {
        med <- median(df_model[[col]], na.rm = TRUE)
        df_model[[col]][is.na(df_model[[col]])] <- med
        dt_model[[col]][is.na(dt_model[[col]])] <- med
    }
}

# Fill missing categorical with mode
for (col in names(df_model)) {
    if (is.factor(df_model[[col]])) {
        mode_val <- names(table(df_model[[col]]))[
            which.max(table(df_model[[col]]))
        ]
        df_model[[col]][is.na(df_model[[col]])] <- mode_val
        if (col %in% names(dt_model)) {
            dt_model[[col]][is.na(dt_model[[col]])] <- mode_val
        }
    }
}

missing_train <- sum(is.na(df_model))
missing_test <- sum(is.na(dt_model))
cat("  Missing values in training data:", missing_train, "\n")
cat("  Missing values in test data:", missing_test, "\n\n")

# ============================================================
# ONE-HOT ENCODING FOR TREE MODELS
# ============================================================
cat("Preparing one-hot encoded versions...\n")

dummies_recipe <- recipe(~., data = df_model %>% select(-all_of(TARGET_VAR))) %>%
    step_dummy(all_nominal(), one_hot = TRUE) %>%
    prep(training = df_model %>% select(-all_of(TARGET_VAR)))

df_ohe <- bake(dummies_recipe, new_data = df_model %>% select(-all_of(TARGET_VAR)))
df_ohe[[TARGET_VAR]] <- df_model[[TARGET_VAR]]

dt_ohe <- bake(dummies_recipe, new_data = dt_model)

cat("  One-hot encoded training features:", ncol(df_ohe) - 1, "\n")
cat("  One-hot encoded test features:", ncol(dt_ohe), "\n\n")

# ============================================================
# SCALING FOR LINEAR MODELS
# ============================================================
cat("Preparing scaled versions...\n")

scale_recipe <- recipe(~., data = df_model %>% select(-all_of(TARGET_VAR))) %>%
    step_dummy(all_nominal(), one_hot = TRUE) %>%
    step_normalize(all_numeric()) %>%
    prep(training = df_model %>% select(-all_of(TARGET_VAR)))

df_scaled <- bake(scale_recipe, new_data = df_model %>% select(-all_of(TARGET_VAR)))
df_scaled[[TARGET_VAR]] <- df_model[[TARGET_VAR]]

dt_scaled <- bake(scale_recipe, new_data = dt_model)

cat("  Scaled training features:", ncol(df_scaled) - 1, "\n")
cat("  Scaled test features:", ncol(dt_scaled), "\n\n")

cat("✓ 05_preprocessing.r complete\n")
cat("  Objects: df_model, dt_model, df_ohe, dt_ohe, df_scaled, dt_scaled\n\n")
