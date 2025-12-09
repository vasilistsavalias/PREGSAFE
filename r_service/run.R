# run.R
# This script starts the Plumber API.

# Load the plumber library
library(plumber)

# Print current working directory and file list for debugging on startup
print(paste("Current working directory:", getwd()))
print("Files in current directory:")
print(list.files())

# Check if main.R exists
if (!file.exists("main.R")) {
  stop("CRITICAL ERROR: 'main.R' not found in the working directory!")
}

# --- Dynamic Port for Render ---
# Render sets the PORT environment variable
port_env <- Sys.getenv("PORT")
if (port_env == "") {
  port <- 8000 # Default for local dev
  print("PORT env var not found. Defaulting to 8000.")
} else {
  port <- as.numeric(port_env)
  print(paste("Detected PORT env var:", port))
}

# Plumb the main.R file using the modern plumber::pr()
# This parses the #* annotations in main.R
pr <- plumber::pr("main.R")

# Run the API
# 'swagger=TRUE' enables the Swagger UI documentation at /__docs__/
print(paste("Starting Plumber API on 0.0.0.0:", port))
pr %>% pr_run(host = "0.0.0.0", port = port)