# GDM Pipeline Codebase Documentation (V4 Refactored)

This document provides a detailed explanation of the refactored codebase, which is now split into two distinct pipelines with a shared common library.

---

## 1. Shared Library: `src/gdm_pipeline_common/`

This directory contains all the components, utilities, and configuration logic that is shared between both pipelines.

### Core Logic

- **`config_manager.py`**: Serves as a single source of truth for all configuration parameters from `config.yaml`. It now correctly handles recursive merging of smoke test overrides.
- **`logging_config.py`**: Centralizes and standardizes logging for the entire application.

### Utilities (`utils/`)

- **`common.py`**: Abstracts and centralizes all common file input/output operations (e.g., `save_yaml`, `load_dataset`).
- **`evaluation.py`**: Contains the new, shared quantitative evaluation functions, specifically `pca_eigen_diff` and `convert_numpy_to_native`.

### Components (`components/`)

- **`data_validation.py`**: Performs initial data inspection, cleaning, and outlier handling.
- **`feature_engineering.py`**: Encapsulates all domain-specific feature creation logic.
- **`imputation.py`**: Creates the serializable `ColumnTransformer` for preprocessing (imputation, scaling, etc.).
- **`classical_model_trainer.py`**: Manages the "Train-Real-Test-Real" (TRTR) workflow.
- **`synthesizer_tuner.py`**: Manages the "Train-Synthetic-Test-Real" (TSTR) workflow for baseline synthesizers.
- **`results_analyzer.py`**: Generates plots and analysis from aggregated cross-validation results.
- **`final_model_trainer.py`**: Executes the final workflow for a champion model on the holdout set.
- **`fidelity_analyzer.py`**: Provides qualitative (visual) assessment of synthetic data quality.
- **`quality_analyzer.py`**: **(New)** Provides quantitative assessment of synthetic data quality using SDMetrics and PCA.
- **`optuna_tuner_ctgan.py`**: **(New)** Encapsulates the Optuna study logic for hyperparameter tuning of the CTGAN synthesizer.
- **`optuna_tuner_tvae.py`**: **(New)** Encapsulates the Optuna study logic for hyperparameter tuning of the TVAE synthesizer.
- **`sdv_wrapper.py`**: Wrappers for the baseline CTGAN and TVAE synthesizers.

---

## 2. The Original 4-Stage Pipeline

This pipeline is designed for a full research workflow starting from a single raw dataset.

- **Orchestrator:** `main_original.py`
- **Code:** `src/gdm_pipeline_original/`

### Stages

- **`stage_01_data_preparation.py`**: Takes the raw dataset and performs cleaning, feature engineering, splitting, and cross-validation fold creation.
- **`stage_02_cross_validation.py`**: Systematically evaluates synthesizers and ML models across the 10 folds.
- **`stage_03_results_analysis.py`**: Aggregates results from all folds and identifies champion models.
- **`stage_04_final_evaluation.py`**: Performs a final, unbiased evaluation on the holdout set. **This stage has been enhanced** to now also run the new `QualityAnalyzer` component, producing a `quantitative_quality_report.yaml` with SDMetrics and PCA scores.

---

## 3. The Dimitris Pipeline (Multi-Synthesizer Experimentation)

This pipeline is a powerful, multi-model workflow for processing pre-folded external datasets. It has been significantly enhanced to support parallel tuning and evaluation of multiple synthesizers.

- **Orchestrator:** `main_dimitris.py`
- **Code:** `src/gdm_pipeline_dimitris/`

### Shared Components (New & Refactored)

- **`optuna_tuner_tvae.py`**: A dedicated component that encapsulates the Optuna study logic for finding the best hyperparameters for the **TVAE** model.
- **`optuna_tuner_ctgan.py`**: A new, parallel component for finding the best hyperparameters for the **CTGAN** model.

### Stages (controlled by flags in `main_dimitris.py`)

- **Optuna Tuning & Generation (`--tune` flag):**
  - **`stage_dimitris_optuna_tuning.py`**: This is now the primary generation stage. It loops through each external dataset and, for each one, performs the following for **both CTGAN and TVAE**:
    1. Runs a full Optuna hyperparameter tuning study using the appropriate tuner component (`OptunaTunerTVAE` or `OptunaTunerCTGAN`).
    2. Trains the synthesizer with the best-found parameters.
    3. Saves the resulting synthetic data to `outputs/dimitris_experiment_optuna/{dataset_name}/`.
    4. Saves a summary of the best hyperparameters found for each model and dataset.

- **Evaluation (`--evaluate` flag):**
  - **`stage_dimitris_evaluation.py`**: This stage has been massively upgraded. It is pointed at an output directory (e.g., `outputs/dimitris_experiment_optuna/`) and performs a comprehensive evaluation for every synthesizer it finds data for (CTGAN and TVAE). For each model in each dataset folder, it now generates:
    1. **SDV Quality Report (`.pkl`)**: The raw, interactive SDV report object.
    2. **Human-Readable SDV Report (`.yaml`)**: A detailed, structured YAML file containing the overall score, property scores, and the granular results from all statistical tests (KS-Test, etc.).
    3. **Enhanced TSTR Metrics**: An aggregated CSV file containing a richer set of machine learning utility scores, including **MCC** and **F2-Score**.
    4. **Fidelity Plots**: A full suite of visual plots (PCA, correlation, etc.) to qualitatively assess data fidelity.
    5. **New Output Structure**: All SDV reports are now saved into a clean, hierarchical folder: `{dataset_name}/sdv_reports/{synthesizer}/`.

---

## 4. Workspace Cleanup

To remove all generated artifacts (logs, outputs, caches, etc.) and return the repository to a clean state, you can use the provided cleaning scripts.

```bash
# For Linux/macOS or Git Bash on Windows
bash scripts/clean.sh

# For Windows PowerShell
powershell -ExecutionPolicy Bypass -File ./scripts/clean.ps1
```
