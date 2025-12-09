# prediction_logic.R
library(caret)
library(logger)
library(xgboost)
library(randomForest)
library(e1071)
library(naivebayes)

PREGSAFE_prediction_sample <- function(maternal_age, Wt_pre, height, Wgain, GA_weeks, Conception_ART, BMI_pre = NULL,
                                       SDG_option = "smote", algorithm_option = "RF_model", models_path = "models") {
  
  # 1. Load the specific model object
  model_wanted <- get_model_for_prediction(SDG_option, algorithm_option, models_path = models_path)
  
  # 2. Pre-process Inputs
  
  # Calculate GA days
  GA_days <- as.numeric(GA_weeks) * 7
  
  # Calculate or use provided BMI
  if (is.null(BMI_pre) || BMI_pre == 0 || is.na(BMI_pre)) {
    # Guard against division by zero
    if (height > 0) {
        h_m <- height / 100
        BMI_pre <- round(Wt_pre / (h_m^2), 1)
    } else {
        BMI_pre <- 0
    }
  }
  
  # 3. Construct Dataframe
  # CRITICAL: Column names must match exactly what the trained models expect.
  # Based on inspection: "MA", "Conception.ART.01", "Wt.pre", "BMI.pre", "Wgain", "GA.days"
  
  new_samples <- data.frame(
    MA = as.numeric(maternal_age),
    Conception.ART.01 = as.numeric(Conception_ART), # 0 or 1
    Wt.pre = as.numeric(Wt_pre),
    BMI.pre = as.numeric(BMI_pre),
    Wgain = as.numeric(Wgain),
    GA.days = as.numeric(GA_days),
    stringsAsFactors = FALSE
  )
  
  # 4. Model-Specific Handling
  
  prediction <- NULL
  prob_value <- 0.0
  
  # XGBoost often requires a matrix
  if (algorithm_option == "XGB_model") {
      # Ensure column order matches the model's feature names if possible
      # But usually caret handles this if names match.
      # For safety with XGBoost (which can be finicky about 'matrix' vs 'df'), we try standard predict first.
      
      # Note: The 'res_ML' objects are 'train' objects from caret.
      # Caret's predict() handles the internal matrix conversion usually.
      
      prediction <- predict(model_wanted, newdata = new_samples)
      probs <- predict(model_wanted, newdata = new_samples, type = "prob")
      
  } else {
      # RF, KNN, SVM, NB
      prediction <- predict(model_wanted, newdata = new_samples)
      probs <- predict(model_wanted, newdata = new_samples, type = "prob")
  }

  # 5. Format Output
  
  # Map "neg"/"pos" to user-friendly text
  prediction_text <- ifelse(as.character(prediction) == "neg", "GDM Negative", "GDM positive")
  
  # Extract Probability of Positive
  if (!is.null(probs)) {
    if (is.matrix(probs) || is.data.frame(probs)) {
      if ("pos" %in% colnames(probs)) {
        prob_value <- probs[, "pos"]
      } else if (ncol(probs) >= 2) {
        prob_value <- probs[, 2] # Fallback to 2nd column
      }
    } else {
      # If it's a vector
      prob_value <- as.numeric(probs[2])
    }
  }
  
  if (is.na(prob_value)) prob_value <- 0.0

  return(list(Predicted = prediction_text, `Probability of GDM positive` = prob_value))
}
