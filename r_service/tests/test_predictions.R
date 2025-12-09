# test_predictions.R
library(testthat)

# Source the logic and model loader to be tested
source("../model_loader.R")
source("../prediction_logic.R")

# --- Test Suite for Prediction Logic ---

test_that("Prediction function runs without errors and returns correct structure", {
  
  # 1. Setup: Load the actual models and features
  all_models <- load_all_models("../models")
  
  # Expect that models have been loaded
  expect_true(length(all_models) > 1, "Models list should not be empty.")
  expect_true("features" %in% names(all_models), "Features list should be loaded.")
  
  # 2. Define a sample input
  maternal_age <- 35
  Wt_pre <- 70
  height <- 165
  Wgain <- 12
  GA_days <- 200
  SDG_option <- "smote"
  algorithm_option <- "RF_model"
  
  # 3. Execution: Call the prediction function
  prediction_output <- PREGSAFE_prediction_sample(
    all_models = all_models,
    maternal_age = maternal_age,
    Wt_pre = Wt_pre,
    height = height,
    Wgain = Wgain,
    GA_days = GA_days,
    SDG_option = SDG_option,
    algorithm_option = algorithm_option
  )
  
  # 4. Assertions: Check the output
  # Check that the output is a list
  expect_type(prediction_output, "list")
  
  # Check that the list has the two expected named elements
  expect_named(prediction_output, c("Predicted", "Probability of GDM positive"))
  
  # Check that the probability is a single numeric value between 0 and 1
  expect_length(prediction_output$`Probability of GDM positive`, 1)
  expect_type(prediction_output$`Probability of GDM positive`, "double")
  expect_gte(prediction_output$`Probability of GDM positive`, 0)
  expect_lte(prediction_output$`Probability of GDM positive`, 1)
  
  # Check that the prediction is a single character string
  expect_length(prediction_output$Predicted, 1)
  expect_type(prediction_output$Predicted, "character")
  
})

# To run these tests, you would typically use:
# test_file("r_service/tests/test_predictions.R")
