# R Model Integration Guide

This guide provides instructions for data scientists and ML engineers on how to replace the mock R service with a real, production-ready prediction model.

---

## 1. Overview

The `r_service` is a Plumber API designed to serve predictions from models developed in R. The current implementation uses a simple mock endpoint that returns a hardcoded prediction. The following steps will guide you through replacing this mock logic with your trained model.

## 2. Prerequisites

- A trained and saved R model artifact (e.g., `smote_model.rds`).
- A clear understanding of the input data schema your model expects (e.g., column names, data types).

---

## 3. Integration Steps

### Step 3.1: Place the Model Artifact

1. Create a new directory named `models` inside the `r_service` directory.
2. Copy your saved model artifact (e.g., `smote_model.rds`) into this new `r_service/models/` directory.

This keeps the project structure clean and separates the model artifacts from the API logic.

### Step 3.2: Modify `plumber.R` to Load the Model

At the top of the `r_service/plumber.R` script, add the code to load your model into memory when the API starts.

**Example:**

```R
# plumber.R

# Load the trained model at startup
# This ensures the model is loaded only once, not on every request.
model <- readRDS("models/smote_model.rds")

#* @apiTitle GDM SMOTE Prediction API
#* @apiDescription An API for generating predictions using SMOTE-based models.

# ... (rest of the file)
```

### Step 3.3: Update the Prediction Endpoint

Modify the `/predict` function to use your loaded model instead of the mock logic.

1. **Preprocess the Input:** The incoming data (`req$body`) will be a list or data frame. You may need to preprocess it to match the exact format your model's `predict()` function expects (e.g., select specific columns, convert data types).
2. **Call the Predict Function:** Use the `predict()` function with your loaded `model` and the preprocessed data.
3. **Format the Output:** Ensure the function returns a list with the prediction and the model name, just like the mock service.

**Example:**

```R
#* Predict GDM risk
#* This endpoint now uses a real, loaded SMOTE-based model.
#* @parser json
#* @post /predict
function(req, res) {
  # The @parser json annotation automatically parses the JSON body.
  data <- req$body

  # 1. Preprocess the incoming data to match the model's expected format.
  #    This is a critical step. The column names and order must be correct.
  #    (This example assumes the model expects a data.frame)
  input_df <- as.data.frame(data)

  # 2. Use the loaded model to make a prediction.
  #    The exact syntax may vary based on your model type (e.g., randomForest, glm).
  prediction_raw <- predict(model, newdata = input_df, type = "response") # Or type = "class"

  # 3. Format the prediction into a user-friendly string.
  #    (This example assumes the model returns 1 for Positive, 0 for Negative)
  prediction_result <- ifelse(prediction_raw > 0.5, "GDM Positive", "GDM Negative")

  # 4. Return the final response.
  list(
    prediction = prediction_result,
    model_used = "Live SMOTE Model" # Update the model name
  )
}
```

---

## 4. Final Verification

After making these changes, restart the R service (`Rscript main.R`). The API will now be serving live predictions from your trained model. You can test it using the application's frontend by selecting the "SMOTE" option.
