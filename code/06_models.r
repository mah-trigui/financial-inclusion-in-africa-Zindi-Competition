# ============================================================
# 06_models.r
# Project : Financial Inclusion in Africa — Bank Account Prediction
# Purpose : Train multiple classification models:
#           Random Forest, Ranger, XGBoost, GLM, SVM, Naive Bayes, etc.
# Requires: df_model, dt_model, df_ohe, dt_ohe, df_scaled, dt_scaled
# Outputs : model_predictions (named list of predictions)
# ============================================================

cat("\n========== 06: MODELS ==========\n\n")

# ============================================================
# PREPARE DATA FOR MODELING
# ============================================================
cat("Preparing data for modeling...\n")

# Convert target to numeric (0/1) for some models
y_train <- as.numeric(df_model[[TARGET_VAR]] == "Yes")
y_test <- rep(NA, nrow(dt_model))

model_predictions <- data.frame(
    uniqueid = dtu[[TEST_ID_VAR]],
    country = dtu$country
)

cat("\n")

# ============================================================
# MODEL 1: RANDOM FOREST
# ============================================================
cat("Training Random Forest...\n")

tryCatch({
    set.seed(MODEL_SEED)
    rf_model <- randomForest(
        x = df_model %>% select(-all_of(TARGET_VAR)),
        y = factor(df_model[[TARGET_VAR]]),
        ntree = RANDOM_FOREST_NTREE,
        importance = TRUE,
        prob = TRUE
    )
    rf_pred <- predict(rf_model, dt_model, type = "prob")[, "Yes"]
    model_predictions$rf <- rf_pred
    cat("  ✓ Random Forest predictions generated\n")
}, error = function(e) cat("  ✗ Random Forest failed:", e$message, "\n"))

# ============================================================
# MODEL 2: RANGER
# ============================================================
cat("Training Ranger (fast Random Forest)...\n")

tryCatch({
    set.seed(MODEL_SEED)
    ranger_model <- ranger(
        formula = as.formula(paste(TARGET_VAR, "~ .")),
        data = df_model,
        num.trees = RANDOM_FOREST_NTREE,
        probability = TRUE,
        respect.unordered.factors = TRUE
    )
    ranger_pred <- predict(ranger_model, dt_model)$predictions[, "Yes"]
    model_predictions$ranger <- ranger_pred
    cat("  ✓ Ranger predictions generated\n")
}, error = function(e) cat("  ✗ Ranger failed:", e$message, "\n"))

# ============================================================
# MODEL 3: XGBOOST
# ============================================================
cat("Training XGBoost...\n")

tryCatch({
    set.seed(MODEL_SEED)

    X_train <- as.matrix(df_ohe %>% select(-all_of(TARGET_VAR)))
    X_test <- as.matrix(dt_ohe)

    xgb_model <- xgboost(
        data = X_train,
        label = y_train,
        nrounds = XGBOOST_NROUNDS,
        max_depth = 6,
        eta = 0.1,
        objective = "binary:logistic",
        verbose = 0
    )
    xgb_pred <- predict(xgb_model, X_test)
    model_predictions$xgboost <- xgb_pred
    cat("  ✓ XGBoost predictions generated\n")
}, error = function(e) cat("  ✗ XGBoost failed:", e$message, "\n"))

# ============================================================
# MODEL 4: LOGISTIC REGRESSION
# ============================================================
cat("Training Logistic Regression...\n")

tryCatch({
    set.seed(MODEL_SEED)

    glm_model <- glm(
        formula = as.formula(paste(TARGET_VAR, "~ .")),
        data = df_model,
        family = binomial(link = "logit")
    )
    glm_pred <- predict(glm_model, dt_model, type = "response")
    model_predictions$glm <- as.numeric(glm_pred)
    cat("  ✓ Logistic Regression predictions generated\n")
}, error = function(e) cat("  ✗ Logistic Regression failed:", e$message, "\n"))

# ============================================================
# MODEL 5: SVM
# ============================================================
cat("Training Support Vector Machine...\n")

tryCatch({
    set.seed(MODEL_SEED)

    svm_model <- svm(
        formula = as.formula(paste(TARGET_VAR, "~ .")),
        data = df_model,
        kernel = "radial",
        cost = 5,
        probability = TRUE
    )
    svm_pred <- attr(
        predict(svm_model, dt_model, probability = TRUE),
        "probabilities"
    )[, "Yes"]
    model_predictions$svm <- as.numeric(svm_pred)
    cat("  ✓ SVM predictions generated\n")
}, error = function(e) cat("  ✗ SVM failed:", e$message, "\n"))

# ============================================================
# MODEL 6: NAIVE BAYES
# ============================================================
cat("Training Naive Bayes...\n")

tryCatch({
    set.seed(MODEL_SEED)

    nb_model <- naivebayes::naive_bayes(
        formula = as.formula(paste(TARGET_VAR, "~ .")),
        data = df_model
    )
    nb_pred <- predict(nb_model, dt_model, type = "prob")[, "Yes"]
    model_predictions$nb <- as.numeric(nb_pred)
    cat("  ✓ Naive Bayes predictions generated\n")
}, error = function(e) cat("  ✗ Naive Bayes failed:", e$message, "\n"))

# ============================================================
# MODEL 7: GRADIENT BOOSTING MACHINE (GBM)
# ============================================================
cat("Training Gradient Boosting Machine...\n")

tryCatch({
    set.seed(MODEL_SEED)

    gbm_model <- gbm(
        formula = as.formula(paste(TARGET_VAR, "~ .")),
        data = df_model,
        distribution = "bernoulli",
        n.trees = 500,
        interaction.depth = 5,
        shrinkage = 0.1,
        verbose = FALSE
    )
    gbm_pred <- predict(gbm_model, dt_model, n.trees = 500, type = "response")
    model_predictions$gbm <- as.numeric(gbm_pred)
    cat("  ✓ GBM predictions generated\n")
}, error = function(e) cat("  ✗ GBM failed:", e$message, "\n"))

cat("\n✓ 06_models.r complete\n")
cat("  Models trained:", ncol(model_predictions) - 2, "\n")
cat("  Object: model_predictions\n\n")
