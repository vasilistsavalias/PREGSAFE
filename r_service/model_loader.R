# model_loader.R
library(logger)

.model_cache <- new.env(parent = emptyenv())

# Loads a single .Rda file which must contain 'res_ML' list
load_model_file <- function(model_file_name, models_path = "models") {
  # Normalize key (e.g., "smote_model" -> "smote_model")
  key <- sub("\\.Rda$", "", model_file_name, ignore.case = TRUE)
  
  if (exists(key, envir = .model_cache, inherits = FALSE)) {
    log_info("Using cached model suite: {key}")
    return(get(key, envir = .model_cache))
  }

  rda_path <- file.path(models_path, paste0(key, ".Rda"))
  
  if (!file.exists(rda_path)) {
    stop("Model file not found: ", rda_path)
  }

  log_info("Loading model suite: {rda_path}")
  
  # MEMORY OPTIMIZATION: Single-Slot Cache
  # Before loading a new suite, clear the previous one to prevent OOM.
  rm(list = ls(envir = .model_cache), envir = .model_cache)
  gc() # Force garbage collection immediately
  
  # Load into a temporary environment to inspect contents
  env <- new.env(parent = emptyenv())
  loaded_vars <- load(rda_path, envir = env)
  
  if (!("res_ML" %in% loaded_vars)) {
    stop("The .Rda file '", rda_path, "' does not contain the expected 'res_ML' object. Found: ", paste(loaded_vars, collapse=", "))
  }

  model_list <- env$res_ML
  
  # Cache the entire list of 5 models
  assign(key, model_list, envir = .model_cache)
  log_info("Cached model suite: {key} with algorithms: {paste(names(model_list), collapse=', ')}")
  
  return(model_list)
}

# Retrieves a specific algorithm (e.g., "XGB_model") from a specific SDG suite (e.g., "smote")
get_model_for_prediction <- function(SDG_option = "smote", algorithm_option = "RF_model", models_path = "models") {
  # Construct the filename key, e.g., "smote_model" or "GAN_model"
  # The frontend sends "smote", we need "smote_model"
  # Exception: if frontend sends "Original", file is "Original_model"
  
  # Basic mapping or appending "_model"
  if (SDG_option == "CTGAN") {
      file_key <- "GAN_model"
  } else if (!grepl("_model$", SDG_option)) {
    file_key <- paste0(SDG_option, "_model")
  } else {
    file_key <- SDG_option
  }
  
  model_list <- load_model_file(file_key, models_path = models_path)
  
  if (is.null(model_list[[algorithm_option]])) {
    stop("Algorithm '", algorithm_option, "' not found in model suite '", file_key, "'. Available: ", paste(names(model_list), collapse=", "))
  }
  
  return(model_list[[algorithm_option]])
}

# Preload a default model to warm up the cache (optional)
preload_default_model <- function(models_path = "models") {
  tryCatch({
    load_model_file("smote_model", models_path = models_path)
  }, error = function(e) {
    log_warn("preload_default_model failed: {conditionMessage(e)}")
  })
}