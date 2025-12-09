# GCP Server: Full Experimental Run Guide

This guide provides instructions for connecting to and running experiments on a Google Cloud VM.

---

## Step 1: Connect to Your VM

Use the `gcloud` CLI from your local machine to establish an SSH connection.

```bash
# Set your project (if not already configured)
gcloud config set project YOUR_PROJECT_ID

# Connect to the VM
gcloud compute ssh YOUR_INSTANCE_NAME --zone=YOUR_ZONE
```

**Example:**

```bash
gcloud config set project ml-research-prod-2024
gcloud compute ssh gpu-instance-2024-01-15 --zone=us-west1-b 

You are now operating inside the VM's terminal.

---

## Step 2: Prepare the Server Environment

### 2.1 Install System Dependencies

Update the server's package list and install necessary tools.

```bash
# Update package lists
sudo apt-get update

# Install Git, Python venv, and compression tools
sudo apt-get install -y git python3-venv zip

# Optional: Install tmux for persistent sessions and htop for monitoring
sudo apt-get install -y tmux htop
```

### 2.2 Set Up SSH for Private Git Repository

To clone a private repository, create an SSH key on the server and add it to your Git provider.

**Generate the SSH key:**

```bash
ssh-keygen -t ed25519 -C "YOUR_KEY_DESCRIPTION"
```

**Example:**

```bash
ssh-keygen -t ed25519 -C "ml-server-deploy-key"
```

Press Enter three times to accept defaults (no passphrase).

**Display the public key:**

```bash
cat ~/.ssh/id_ed25429.pub
```

**Add the key to your repository:**

1. Copy the entire output (starting with `ssh-ed25429...`)
2. Navigate to your repository settings
3. Go to **Settings > Deploy Keys > Add deploy key**
4. Paste the key and give it a title (e.g., "GCP Server Deploy Key")
5. Do not enable write access
6. Click "Add key"

### 2.3 Clone the Repository

Clone using the SSH URL (not HTTPS).

```bash
# Clone the repository
git clone git@github.com:USERNAME/REPOSITORY.git

# Navigate into the project
cd REPOSITORY
```

**Example:**

```bash
git clone git@github.com:ml-research-team/medical-llm-pipeline.git
cd medical-llm-pipeline
```

**For subsequent runs**, use these commands to sync and switch branches:

```bash
# Switch branches (if needed)
git checkout BRANCH_NAME

# Discard local changes and sync with remote
git reset
git pull

# View recent commit history (optional)
git log -p -n 1
```

**Example:**

```bash
git checkout experimental-v2
git reset
git pull
```

### 2.4 Workspace Cleanup

Remove generated artifacts (logs, outputs, caches) to return to a clean state.

**On Linux/macOS or Git Bash:**

```bash
bash ml_pipeline/scripts/clean.sh
```

**On Windows PowerShell:**

```powershell
powershell -ExecutionPolicy Bypass -File ./ml_pipeline/scripts/clean.ps1
```

---

## Step 3: Set Up Python Environment

This is a one-time setup. The run scripts will automatically activate the environment for you in the future.

### 3.1 Create and Activate Virtual Environment

This isolates your project's dependencies and is required by the server's OS.

```bash
# Navigate to project root (if not already there)
cd REPOSITORY_NAME

# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 3.2 Install All Dependencies

With the virtual environment active, install all required packages.

```bash
# Install backend and ML pipeline requirements
pip install -r backend/requirements.txt
pip install -r ml_pipeline/requirements.txt

# CRITICAL: Install the local ml_pipeline package in "editable" mode
# This makes your local modules (e.g., gdm_pipeline_common) importable
pip install -e ml_pipeline
```

**Example:**

```bash
cd medical-llm-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
pip install -r ml_pipeline/requirements.txt
pip install -e ml_pipeline
```

Your environment is now fully configured. You can proceed to run the pipelines.

---

## Step 4: Run the Pipeline

We now have a unified Master Pipeline script that handles everything.

### 4.1 Make Scripts Executable

```bash
chmod +x ml_pipeline/scripts/run_pipeline.sh
```

### 4.2 Standard Production Run

This runs the baseline generation (no expensive tuning) for all folds and trains the final model.

```bash
./ml_pipeline/scripts/run_pipeline.sh
```

### 4.3 Full Tuning Run (Recommended)

This runs Optuna hyperparameter tuning for all 10 folds before generation. This is computationally expensive but yields better results. **If the run crashes, re-running this command will resume tuning from the last checkpoint.**

```bash
./ml_pipeline/scripts/run_pipeline.sh --tune
```

### Optional: Run in Persistent Session

For long-running experiments, use `tmux` to keep the process running even if you disconnect:

```bash
# Start a new tmux session
tmux new -s gdm_experiment

# Run your pipeline inside tmux
./ml_pipeline/scripts/run_pipeline.sh --tune

# Detach from session: Press Ctrl+B, then D
# Reattach later: tmux attach -t gdm_experiment
# List sessions: tmux ls
```

---

## Step 5: Package and Download Results

### 5.1 Create Archive on VM

Once the pipeline completes, package all output files into a compressed archive.

```bash
tar -czvf /home/USERNAME/gdm_results.tar.gz -C ml_pipeline/outputs .
```

**Example:**

```bash
tar -czvf /home/jsmith/experiment_results.tar.gz -C ml_pipeline/outputs baseline_run_2024
```

**Command breakdown:**

| Flag | Description |
|------|-------------|
| `-c` | Create a new archive |
| `-z` | Compress using gzip |
| `-v` | Verbose output (show files being added) |
| `-f` | Specify output filename |
| `-C` | Change to directory before archiving |

**Result:** Creates `/home/USERNAME/RESULTS_NAME.tar.gz` containing all files from `ml_pipeline/outputs/EXPERIMENT_FOLDER/`.

### 5.2 Download to Local Machine

Exit the VM session:

```bash
exit
```

From your local terminal, download the archive using `gcloud compute scp`:

```bash
gcloud compute scp INSTANCE_NAME:/home/USERNAME/RESULTS_NAME.tar.gz "LOCAL_DESTINATION_PATH" --zone=YOUR_ZONE
```

**Example (Linux/macOS):**

```bash
gcloud compute scp gpu-instance-2024-01-15:/home/jsmith/experiment_results.tar.gz ~/Downloads --zone=us-west1-b
```

**Example (Windows):**

```bash
gcloud compute scp gpu-instance-2024-01-15:/home/jsmith/experiment_results.tar.gz "C:\Users\JSmith\Downloads" --zone=us-west1-b
```

**Command breakdown:**

| Component | Description |
|-----------|-------------|
| `gcloud compute scp` | Secure copy between local and VM |
| `instance-name:` | VM instance identifier |
| `/path/on/vm` | Source file path on VM |
| `"local/path"` | Destination on local machine |
| `--zone` | VM zone specification |

---

## Step 6: Extract Results Locally

### On Linux/macOS

```bash
cd ~/Downloads
mkdir gdm_results
tar -xzvf gdm_results.tar.gz -C gdm_results
```

### On Windows

```powershell
cd "%USERPROFILE%\Downloads"
mkdir gdm_results
tar -xzvf gdm_results.tar.gz -C gdm_results
```

**Extraction flags:**

| Flag | Description |
|------|-------------|
| `-x` | Extract files from archive |
| `-z` | Decompress gzip compression |
| `-v` | Verbose output (list extracted files) |
| `-f` | Specify archive filename |
| `-C` | Extract to specific directory |

**Result:** All experiment outputs will appear in the `gdm_results/` directory.

---

## Quick Reference

### Common Git Commands

```bash
# Check status
git status

# Pull latest changes
git pull

# Switch branches
git checkout BRANCH_NAME

# View commit history
git log --oneline -n 10
```

### Common VM Commands

```bash
# Check disk space
df -h

# Check running processes
ps aux | grep python

# Monitor resource usage (if htop installed)
htop

# Monitor resource usage (alternative)
top
```

### Tmux Session Management

```bash
# Create new session
tmux new -s session_name

# List all sessions
tmux ls

# Attach to existing session
tmux attach -t session_name

# Detach from session
# Press: Ctrl+B, then D

# Kill a session
tmux kill-session -t session_name
```

### Reconnect to VM

```bash
gcloud compute ssh YOUR_INSTANCE_NAME --zone=YOUR_ZONE
```

**Example:**

```bash
gcloud compute ssh gpu-instance-2024-01-15 --zone=us-west1-b
```
