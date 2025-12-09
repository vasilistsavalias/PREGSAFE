# PREGSAFE

## Public Release Note

This repository is a sanitized public release of the PREGSAFE platform. Sensitive medical datasets,commits and internal research logs have been excluded for privacy compliance.

## 1. Project Overview

This repository contains a full-stack MLOps platform for predicting the risk of Gestational Diabetes Mellitus (GDM). The project is architected as a modern web application with a clear separation of concerns between the user interface, the prediction API, and the underlying machine learning research environment.

The platform consists of three primary, decoupled runtime components:

1. **Frontend Application:** A user-friendly web interface built with Next.js and React that allows clinicians to input patient data and receive real-time risk predictions.
2. **Backend Service (Python):** A high-performance REST API built with Python and FastAPI. It orchestrates predictions, serving as a secure **proxy** to the R Service for all model inferences (including the primary CTGAN model).
3. **R Service:** An API built with R and Plumber, responsible for serving predictions from the complete suite of 40+ ML models (SMOTE, CTGAN, etc.).

Separately, the project includes a comprehensive **ML Research Pipeline** for model training, evaluation, and artifact generation.

---

## 2. Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | Next.js, React, TypeScript, Tailwind CSS |
| **Backend** | Python, FastAPI, Uvicorn, SDV, Gower |
| **R Service** | R, Plumber, Docker |
| **ML Pipeline** | Python, Optuna, Pandas, Scikit-learn, SDV |

---

## 3. Project Structure

```gdm
gdm/
├── 0.READMEs/            # Additional architectural and guide documentation
├── backend/              # FastAPI Backend (Python)
├── database/             # SQLite database file
├── frontend/             # Next.js Frontend (React)
├── ml_pipeline/          # Unified ML Pipeline (Folds & Final Model)
├── .gitignore
└── README.md
```

---

## 4. Local Development Environment Setup

To run the complete application locally, you will need to start all three services (Frontend, Backend, and R Service) in separate terminal sessions. The VS Code `restore-terminals.json` file is configured to do this automatically.

### Prerequisites

- **Node.js:** v18 or later recommended.
- **Python:** v3.11 or later recommended.
- **R:** v4.3 or later recommended. Ensure `Rscript` is available in your system's PATH.
- **R Packages:** You must install the required R packages by running the following commands once:

  ```powershell
  Rscript -e "install.packages(c('caret', 'xgboost', 'kernlab', 'naivebayes', 'plumber', 'logger', 'randomForest', 'e1071'), dependencies=TRUE, repos='https://cloud.r-project.org')"
  ```

---

### Step 1: Start the Backend Service (Python/FastAPI)

This server orchestrates machine learning predictions and runs on port **8008**.

```powershell
# From the project root directory, run:
cd backend
uvicorn main:app --reload --port 8008
```

---

### Step 2: Start the Frontend Application (Next.js)

This is the main user interface for the application and runs on port **3000**.

```powershell
# From the project root directory, run:
cd frontend
npm run dev
```

Open your browser and go to `http://localhost:3000`.

---

### Step 3: Start the R Service (R/Plumber)

This service provides the advanced prediction models and runs on port **8000**.

```powershell
# From the project root directory, run:
cd r_service
Rscript -e "plumber::pr('main.R') |> plumber::pr_run(host='0.0.0.0', port=8000)"
```

---

## 5. Modular Documentation

This project is organized into several distinct modules, each with its own detailed README file. These documents provide in-depth information about the architecture, setup, and usage of each component.

### Service Documentation

- **[Backend Documentation](backend/0.README-backend.md):** A comprehensive guide to the FastAPI-based backend, including API endpoints, database schema, and core logic.
- **[Frontend Documentation](frontend/0.README-frontend.md):** A detailed overview of the Next.js frontend, including component structure, state management, and API integration.
- **[R Service Documentation](r_service/0.README-r_service.md):** A guide to the R-based microservice, explaining its architecture, how to run it in Docker, and its API.
- **[ML Pipeline Documentation](ml_pipeline/0.README-ml_pipeline.md):** A complete guide to the machine learning pipeline, covering data preparation, model training, tuning, and evaluation.

### Advanced Guides & Architecture (0.READMEs)

- **[GCP Execution Guide](0.READMEs/gcp_execution_guide.md):** A step-by-step manual for deploying the research environment to Google Cloud, including SSH connection, dependency management, and running full-scale Optuna experiments.
- **[Pipeline Codebase V4](0.READMEs/pipeline.md):** Technical documentation of the refactored V4 codebase, detailing the shared common library, the 4-stage research workflow, and the multi-synthesizer experimental pipelines.
- **[R Model Integration](0.READMEs/r_model_integration_guide.md):** Instructions for ML engineers on how to replace the mock R service endpoints with real, trained `.rds` model artifacts for production inference.
