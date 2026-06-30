# AML weekly schedule — Medicare classifier (every Monday)

Retrain on **`medicare-training/latest/`** parquet after Express Script refresh.

---

## Prerequisites

- [x] Completed command job (datastore mount + `medicare-train-env` with pyarrow)
- [x] Blob `medicare-training/latest/claims.parquet` (Express Script in prod)
- [x] Datastore `claims_medicare_training`
- [x] Compute `claims-compute` (min nodes 0 — schedule will scale up)

**Recommended timing:** Express Script **Sunday** → AML train **Monday 6 AM ET**

---

## Option A — Azure ML Studio (Portal UI)

> **Why you don’t see Schedule on your job**  
> The **Schedule** button appears only on **pipeline job** detail pages — **not** on standalone **command jobs**.  
> Your successful Medicare run was a command job, so **Schedule → Create new schedule** will not show there.  
> Use **Option B (CLI / Cloud Shell)** below, or **Option A2 (pipeline wrapper)**.

### A2 — Studio pipeline (if you want Schedule in the UI)

1. [ml.azure.com](https://ml.azure.com) → **`claims-ml-workspace`**
2. Left menu → **Authoring** → **Pipelines** → **+ Create** (or **Create pipeline job**)
3. Add a **Command** step (drag from components or “Create command job” inside pipeline)
4. Copy your working command-job settings:

| Field | Value |
|-------|--------|
| Code | `train_medicare_classifier.py` |
| Command | `python train_medicare_classifier.py --data_path ${{inputs.training_data}} --model_output ${{outputs.model_output}}` |
| Environment | `medicare-train-env@latest` |
| Compute | `claims-compute` |
| Input `training_data` | `azureml://datastores/claims_medicare_training/paths/latest/` |
| Output `model_output` | Folder · Upload |
| Env var | `MODELS_DIR=/tmp/models` |

5. **Submit** the pipeline → wait for **Completed**
6. Open the **pipeline job** (not the command child alone) → top bar → **Schedule** → **Create new schedule**
7. Set **Monday 6 AM ET** → **Review + Create**
8. Confirm under **Jobs → All schedules**

### A1 — From pipeline job only (original steps)

### 1. Open your successful **pipeline** job

1. [ml.azure.com](https://ml.azure.com) → workspace **`claims-ml-workspace`**
2. **Jobs** → open a **Completed pipeline job** (type = Pipeline, not Command)

### 2. Create schedule

1. Top toolbar → **Schedule** → **Create new schedule**
2. **Basic settings:**

| Field | Value |
|-------|-------|
| **Name** | `medicare-weekly-retrain` |
| **Trigger** | **Recurrence** |
| **Time zone** | `Eastern Standard Time` |
| **Frequency** | **Week** |
| **Interval** | `1` |
| **Weekday** | **Monday** |
| **Time** | `6:00 AM` |
| **Start** | Next Monday (or today) |
| **End** | Leave blank |

3. **Advanced settings** (verify job config):

| Section | Value |
|---------|--------|
| **Environment** | `medicare-train-env:3` |
| **Compute** | `claims-compute` |
| **Inputs → training_data** | `azureml://datastores/claims_medicare_training/paths/latest/` (ReadWrite mount) |
| **Outputs → model_output** | Folder · ReadWrite mount |
| **Command** | `python train_medicare_classifier.py --data_path ${{inputs.training_data}} --model_output ${{outputs.model_output}}` |
| **Env vars** | None (empty — script uses `/tmp/models` via `resolve_models_dir()`) |

4. **Review + Create**

### 3. Manage schedule

**Jobs → All schedules** → `medicare-weekly-retrain`

- **Disable** to pause
- **Update to existing schedule** — after a successful manual job with improved config

### If **Schedule** button is missing

You are on a **command job** — use **Option B (Cloud Shell)** or **Option A2 (pipeline wrapper)** above.

---

## Option B — Azure Cloud Shell (recommended for command jobs)

Cloud Shell starts in `~` with **no repo** — create a folder and YAML files first, then upload the training script.

### 1. Open Cloud Shell

[portal.azure.com](https://portal.azure.com) → top bar **Cloud Shell** → **Bash**

### 2. Paste this entire block (creates folder + YAML)

```bash
mkdir -p ~/medicare-aml-schedule
cd ~/medicare-aml-schedule

cat > medicare-train-pipeline-job.yml << 'EOF'
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
type: pipeline
display_name: medicare-classifier-train-pipeline
experiment_name: Default
description: Single-step pipeline for scheduled Medicare training
jobs:
  train:
    code: .
    command: >-
      python train_medicare_classifier.py
      --data_path ${{inputs.training_data}}
      --model_output ${{outputs.model_output}}
    environment: azureml:medicare-train-env:3
    compute: azureml:claims-compute
    inputs:
      training_data:
        type: uri_folder
        path: azureml://datastores/claims_medicare_training/paths/latest/
        mode: ro_mount
    outputs:
      model_output:
        type: uri_folder
        mode: upload
EOF

cat > medicare-train-command-job.yml << 'EOF'
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command
display_name: medicare-classifier-train
experiment_name: Default
description: Train Medicare classifier from claims_medicare_training datastore
code: .
command: >-
  python train_medicare_classifier.py
  --data_path ${{inputs.training_data}}
  --model_output ${{outputs.model_output}}
environment: azureml:medicare-train-env:3
compute: azureml:claims-compute
inputs:
  training_data:
    type: uri_folder
    path: azureml://datastores/claims_medicare_training/paths/latest/
    mode: ro_mount
outputs:
  model_output:
    type: uri_folder
    mode: upload
EOF

cat > medicare-weekly-train-schedule.yml << 'EOF'
$schema: https://azuremlschemas.azureedge.net/latest/schedule.schema.json
name: medicare-weekly-retrain
display_name: Medicare weekly retrain (Monday)
description: Retrain Medicare classifier every Monday 6 AM Eastern.
trigger:
  type: cron
  expression: "0 6 * * 1"
  time_zone: Eastern Standard Time
create_job:
  type: pipeline
  job: ./medicare-train-pipeline-job.yml
EOF

echo "YAML files created in ~/medicare-aml-schedule"
ls -la
```

### 3. Upload the training script

1. Cloud Shell toolbar → **Upload/Download files** → **Upload**
2. Select **`train_medicare_classifier.py`** from your Mac
3. Move it into the folder:

```bash
mv ~/train_medicare_classifier.py ~/medicare-aml-schedule/
ls ~/medicare-aml-schedule/
# must show: medicare-train-pipeline-job.yml  medicare-weekly-train-schedule.yml  train_medicare_classifier.py
```

### 4. Create the schedule

```bash
az extension add -n ml --yes 2>/dev/null || az extension update -n ml

cd ~/medicare-aml-schedule

az ml schedule create \
  -f medicare-weekly-train-schedule.yml \
  -g rg-claims-intelligence \
  -w claims-ml-workspace
```

### 5. Verify

```bash
az ml schedule list -g rg-claims-intelligence -w claims-ml-workspace -o table
```

Studio: **Jobs → All schedules** → `medicare-weekly-retrain`

### 6. Test once now (optional)

```bash
az ml schedule trigger \
  -n medicare-weekly-retrain \
  -g rg-claims-intelligence \
  -w claims-ml-workspace
```

---

## Option C — CLI from local Mac

Files:

- `aml/medicare-train-command-job.yml` — job definition
- `aml/medicare-weekly-train-schedule.yml` — Monday 6 AM ET recurrence

```bash
cd medicare_classifier
az login
az account set --subscription "<your-subscription-id>"

# Optional: test job once
az ml job create \
  -f aml/medicare-train-command-job.yml \
  -g rg-claims-intelligence \
  -w claims-ml-workspace

# Create weekly schedule
az ml schedule create \
  -f aml/medicare-weekly-train-schedule.yml \
  -g rg-claims-intelligence \
  -w claims-ml-workspace
```

List / disable:

```bash
az ml schedule list -g rg-claims-intelligence -w claims-ml-workspace -o table
az ml schedule disable -n medicare-weekly-retrain -g rg-claims-intelligence -w claims-ml-workspace
```

---

## Cron alternative (CLI only)

Monday **6:00 AM Eastern** in schedule YAML:

```yaml
trigger:
  type: cron
  expression: "0 6 * * 1"
  time_zone: Eastern Standard Time
```

---

## After each scheduled run

1. **Jobs** → find run under experiment `medicare-weekly-retrain`
2. Check logs: `Loaded N training records` · `Saved ...medicare_classifier.pkl`
3. **Outputs** → `model_output/` → download or promote to repo
4. (TBD) Promotion gate → git push → CD → Container Apps

---

## Checklist

- [ ] Schedule name: `medicare-weekly-retrain`
- [ ] Every **Monday** 6 AM ET (after Sunday Express Script)
- [ ] Environment: `medicare-train-env@latest` (or pinned v3+)
- [ ] Input: `claims_medicare_training` / `latest/`
- [ ] Output: `model_output`
- [ ] Visible under **Jobs → All schedules**
