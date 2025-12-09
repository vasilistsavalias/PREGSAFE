# main.R
library(plumber)
library(logger)
library(jsonlite)

# Load helper scripts
source("model_loader.R")
source("prediction_logic.R")

# Configure Logging
log_formatter(formatter_json)
log_layout(layout_json())

# Preload models (Optional - warm up cache)
preload_default_model()

#* @apiTitle PREGSAFE R Prediction Service
#* @apiDescription R Microservice for serving GDM risk prediction models (SMOTE, GAN, etc.)

#* Health Check
#* @get /health
function() {
  list(status = "online", timestamp = Sys.time())
}

#* Predict GDM Risk
#* @param maternal_age:numeric Maternal Age
#* @param Wt_pre:numeric Pre-pregnancy Weight (kg)
#* @param height:numeric Height (cm)
#* @param Wgain:numeric Weight Gain (kg)
#* @param GA_weeks:numeric Gestational Age (weeks)
#* @param Conception_ART:numeric Conception via ART (0=No, 1=Yes)
#* @param bmi:numeric Calculated BMI (optional)
#* @param SDG_option:character Synthetic Data Gen method (default: 'smote')
#* @param algorithm_option:character Algorithm (default: 'RF_model')
#* @post /predict
function(req, res, 
         maternal_age, Wt_pre, height, Wgain, GA_weeks, Conception_ART, 
         bmi = 0, SDG_option = "smote", algorithm_option = "RF_model") {
  
  # 1. Security Check (API Key)
  api_key <- req$HTTP_X_API_KEY
  
  # 2. Input Parsing & Validation
  tryCatch({
    # Convert inputs to numeric
    ma <- as.numeric(maternal_age)
    wt <- as.numeric(Wt_pre)
    ht <- as.numeric(height)
    wg <- as.numeric(Wgain)
    ga <- as.numeric(GA_weeks)
    art <- as.numeric(Conception_ART)
    bmi_val <- as.numeric(bmi)
    
    if (any(is.na(c(ma, wt, ht, wg, ga)))) {
      res$status <- 400
      return(list(error = "Invalid input: Missing or non-numeric values for required fields."))
    }

    log_info("Request: SDG={SDG_option}, Algo={algorithm_option}, MA={ma}, GA_weeks={ga}, ART={art}")

    # 3. Call Prediction Logic
    result <- PREGSAFE_prediction_sample(
      maternal_age = ma,
      Wt_pre = wt,
      height = ht,
      Wgain = wg,
      GA_weeks = ga,
      Conception_ART = art,
      BMI_pre = bmi_val,
      SDG_option = SDG_option,
      algorithm_option = algorithm_option
    )
    
    return(result)
    
  }, error = function(e) {
    log_error("Prediction failed: {conditionMessage(e)}")
    res$status <- 500
    return(list(error = paste("Prediction Error:", conditionMessage(e))))
  }, finally = {
    # 4. Aggressive Memory Cleanup
    gc()
  })
}
