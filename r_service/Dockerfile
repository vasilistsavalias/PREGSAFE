# Use a specific version of the official Rocker image for reproducibility
FROM rocker/r-ver:4.3.2

# Create /opt/r directory if it's expected by some underlying process
RUN mkdir -p /opt/r

# Install system dependencies
RUN apt-get update -qq && apt-get install -y \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev

# Install R packages
RUN R -e "install.packages(c('plumber', 'logger', 'caret', 'naivebayes', 'xgboost', 'randomForest', 'e1071'), dependencies=TRUE)"

# Set up a non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser

# --- DEBUGGING STEP 1: Log build context ---
# We list the files in the current directory BEFORE copying to see what Docker "sees"
# (This runs on the build machine, outputting to build logs)
# RUN ls -la

# Copy all files from the r_service directory (since context is root)
COPY --chown=appuser:appuser r_service/ .

# --- DEBUGGING STEP 2: Verify file placement ---
# This will print the directory contents to the build log. 
# If run.R is missing here, the COPY failed or context is wrong.
RUN echo "=== Directory Contents after COPY ===" && ls -laR /home/appuser

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Startup command
CMD ["Rscript", "run.R"]