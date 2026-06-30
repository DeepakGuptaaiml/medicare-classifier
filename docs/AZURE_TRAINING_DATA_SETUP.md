# Azure ML training data landing zone

Express Script replicates Oracle `CLAIMS_AI_RO` views into Azure Blob Storage. Azure ML compute (`claims-compute`) reads that data to train models — **no direct Oracle connection from AML**.

## Architecture

```
Oracle (CLAIMS_AI_RO)
        │
        ▼
Express Script (org pipeline, weekly)
        │
        ▼
Blob: medicare-training/  reserve-training/
        │
        ▼
AML datastore (claims_medicare_training, claims_reserve_training)
        │
        ▼
AML command job on claims-compute → train_*.py
```

## Resources (rg-claims-intelligence)

| Resource | Purpose |
|----------|---------|
| `claimsmlstorage4cd105406` | Existing AML workspace storage (default landing account) |
| `medicare-training` | Container — Medicare classifier parquet snapshots |
| `reserve-training` | Container — Reserve forecaster parquet snapshots |
| `claims_medicare_training` | AML datastore → `medicare-training` container |
| `claims_reserve_training` | AML datastore → `reserve-training` container |
| `claims-ml-workspace` | Registers datastores; jobs mount inputs |
| `claims-compute` | CPU cluster runs training jobs |

## Azure Portal setup (step-by-step)

Use this when you prefer the Portal over CLI. All resources live in resource group **`rg-claims-intelligence`**.

### Prerequisites

| Item | Your value |
|------|------------|
| Subscription | Your Azure subscription |
| Resource group | `rg-claims-intelligence` |
| Storage account | `claimsmlstorage4cd105406` (existing AML workspace storage) |
| AML workspace | `claims-ml-workspace` |
| Compute cluster | `claims-compute` |

---

### Step 1 — Create blob containers (training landing zone)

Express Script will write parquet here. AML jobs read from these containers.

1. Open [Azure Portal](https://portal.azure.com).
2. Search **Storage accounts** → open **`claimsmlstorage4cd105406`**.
3. Left menu → **Data storage** → **Containers**.
4. Click **+ Container** and create:

   | Field | Medicare | Reserve |
   |-------|----------|---------|
   | Name | `medicare-training` | `reserve-training` |
   | Public access level | **Private** | **Private** |

5. (Optional) Inside each container, create a virtual folder **`latest/`** by uploading a placeholder file named `latest/.keep` — or let Express Script create `latest/claims.parquet` on first run.

**Expected layout after Express Script runs:**

```
medicare-training/
  latest/
    claims.parquet          (or partitioned *.parquet)

reserve-training/
  latest/
    claims.parquet
```

**Blob URLs for Express Script team:**

```
https://claimsmlstorage4cd105406.blob.core.windows.net/medicare-training/latest/
https://claimsmlstorage4cd105406.blob.core.windows.net/reserve-training/latest/
```

---

### Step 2 — Grant AML workspace access to storage (RBAC)

The workspace managed identity must read blobs when jobs mount the datastore.

#### Option A — Use account key on datastore (no identity needed)

When creating the datastore (Step 4), choose **Authentication type: Account key**. AML uses the storage key; you can skip workspace identity for a first test.

#### Option B — Workspace managed identity (recommended for production)

**Where to find Identity in Portal** (UI varies by region/version):

1. [Azure Portal](https://portal.azure.com) → search **`claims-ml-workspace`**.
2. Open the **Machine learning** resource (not Azure ML Studio).
3. In the left menu, look for **Identity** directly (key icon) — it may be under **Settings**, or listed on its own.
4. **System assigned** tab → set **Status** to **On** → **Save**.
5. Copy the **Object (principal) ID** shown after save.

If you still do not see **Identity**:

- Your role may lack permission to view/edit identity (need **Managed Identity Operator** or **Owner**).
- Try: Portal → **Microsoft Entra ID** → **Enterprise applications** → search `claims-ml-workspace` — if it appears, identity is already on.
- Or use **Option A (account key)** for now.

**Assign storage role to workspace identity:**

1. Storage account **`claimsmlstorage4cd105406`** → **Access control (IAM)**.
2. **+ Add** → **Add role assignment**.
3. Role: **Storage Blob Data Contributor** → **Next**.
4. Members: search **`claims-ml-workspace`** or paste the **Object ID** from step 5 above.
5. **Review + assign**.

---

### Step 3 — Grant Express Script access to storage (RBAC)

Your org’s Express Script service principal (or managed identity) needs to **write** parquet into the containers.

1. On storage account **`claimsmlstorage4cd105406`** → **Access control (IAM)**.
2. **+ Add** → **Add role assignment**.
3. Role: **Storage Blob Data Contributor**.
4. Members: select the **Express Script** app registration, managed identity, or service principal used by that pipeline.
5. **Review + assign**.

(Optional) Scope to a single container instead of the whole account:

1. Open container **`medicare-training`** (not the storage account root).
2. **Access control (IAM)** → add **Storage Blob Data Contributor** for Express Script on that container only.
3. Repeat for **`reserve-training`**.

---

### Step 4 — Register AML datastores

Datastores tell Azure ML where training files live so jobs can mount them.

#### 4a — Medicare training datastore

1. Open **`claims-ml-workspace`** in the Portal.
2. Left menu → **Assets** → **Data** → **Data stores** tab.
3. **+ Create**.
4. Choose **Azure Blob Storage** → **Next**.
5. Fill in:

   | Field | Value |
   |-------|-------|
   | Name | `claims_medicare_training` |
   | Storage account | `claimsmlstorage4cd105406` |
   | Container | `medicare-training` |
   | Authentication | **Account key** (simplest) **or** **Credentials from Azure ML workspace** (uses workspace identity from Step 2) |

6. **Create**.

#### 4b — Reserve training datastore

Repeat with:

| Field | Value |
|-------|-------|
| Name | `claims_reserve_training` |
| Container | `reserve-training` |

---

### Step 5 — Verify in Portal

**Containers**

1. Storage account → **Containers** → open `medicare-training`.
2. After a test upload or Express Script run, you should see `latest/claims.parquet` (or `latest/*.parquet`).

**Datastores**

1. AML workspace → **Data** → **Data stores**.
2. Confirm both **`claims_medicare_training`** and **`claims_reserve_training`** show status **Completed**.

**IAM**

1. Storage account → **Access control (IAM)** → **Role assignments**.
2. Filter by role **Storage Blob Data Contributor**.
3. Confirm **`claims-ml-workspace`** and Express Script principal are listed.

---

### Step 6 — Test upload via Portal (optional)

Prepare parquet from your local CSV:

```bash
cd medicare_classifier
pip install pandas pyarrow
python scripts/csv_to_parquet.py
# creates data/claims.parquet (1,000 rows from claims_data.csv)
```

Upload in Portal:

1. Storage account → **Containers** → **`medicare-training`**.
2. **Upload** → select `data/claims.parquet`.
3. **Advanced** → set blob name to **`latest/claims.parquet`** (folder `latest/` + file name).
4. Upload.

Confirm: container should show folder **`latest`** containing **`claims.parquet`**.

---

### Step 7 — Use datastore in an AML training job (Studio)

#### Your Inputs panel — correct

From your screenshot, this is **correct**:

| Field | Your value | OK? |
|-------|------------|-----|
| Input name | `training_data` | Yes |
| Input type | Data | Yes |
| Data type | Folder | Yes |
| Data source | URI | Yes |
| URI | `azureml://datastores/claims_medicare_training/paths/latest/` | Yes |
| Input mode | Read-only mount | Yes |

#### Fix: `Missing inputs from command: training_data`

Azure ML uses **two different placeholder syntaxes**:

| Where | Syntax | Example |
|-------|--------|---------|
| **Command** field | `$[[inputs.name]]` / `$[[outputs.name]]` | `$[[inputs.training_data]]` |
| **Environment variables** | `${{inputs.name}}` / `${{outputs.name}}` | `${{inputs.training_data}}` |

**`${{outputs.model_output}}` in the Command field is wrong** — that triggers validation errors (sometimes reported as missing `training_data` even when the real bug is output syntax).

---

#### Option A — Shell wrapper (recommended; fixes input + output wiring)

**Outputs** — add this (required for your `cp` step):

| Field | Value |
|-------|-------|
| Name | `model_output` |
| Type | Data → Folder |
| Mode | Upload (or Read-write mount) |

**Command** — use `$[[ ]]` double brackets:

```text
bash scripts/aml_run_training.sh $[[inputs.training_data]] $[[outputs.model_output]]
```

**Inputs** (unchanged):

| Name | URI |
|------|-----|
| `training_data` | `azureml://datastores/claims_medicare_training/paths/latest/` |

**Environment variables** — literals only (no `${{inputs...}}` needed):

| Name | Value |
|------|-------|
| `MODELS_DIR` | `/tmp/models` |

The wrapper sets `TRAINING_DATA_URI` from arg1 and copies artifacts to arg2.

---

#### Option B — One-liner command (no wrapper)

**Outputs:** `model_output` (uri_folder, Upload)

**Command:**

```text
TRAINING_DATA_URI=$[[inputs.training_data]] MODELS_DIR=/tmp/models python train_medicare_classifier.py && cp /tmp/models/medicare_classifier.pkl /tmp/models/preprocess_config.json /tmp/models/model_comparison.csv $[[outputs.model_output]]/
```

Do **not** use `${{outputs.model_output}}` or `${{inputs.training_data}}` in Command.

---

#### Option C — Simplest (no `cp` in command)

`train_medicare_classifier.py` already copies artifacts when AML sets `AZUREML_OUTPUT_MODEL_OUTPUT` (from output name `model_output`).

**Outputs:** `model_output` (uri_folder)

**Command:**

```text
TRAINING_DATA_URI=$[[inputs.training_data]] MODELS_DIR=/tmp/models python train_medicare_classifier.py
```

---

#### Your current command — what to change

**Wrong (what you have now):**

```text
python train_medicare_classifier.py && cp ... ${{outputs.model_output}}/
```

**Right:**

```text
bash scripts/aml_run_training.sh $[[inputs.training_data]] $[[outputs.model_output]]
```

Also confirm **Outputs** includes `model_output` — without it, any `$[[outputs.model_output]]` reference will fail.

**Common mistakes:**

| Mistake | Fix |
|---------|-----|
| `${{outputs.model_output}}` in Command | Use `$[[outputs.model_output]]` |
| `${{inputs.training_data}}` in Command | Use `$[[inputs.training_data]]` or wrapper script |
| Output `model_output` not defined | Add under **Outputs** tab |
| Env var `${{inputs.training_data}}` + complex command | Prefer Option A wrapper; references inputs in Command via `$[[ ]]` |

After submit, in job logs you should see:

```text
Loading training data from /mnt/azureml/inputs/training_data
```

---

### Portal checklist (print-friendly)

- [ ] Container `medicare-training` created (private)
- [ ] Container `reserve-training` created (private)
- [ ] AML workspace system-assigned identity **On**
- [ ] `claims-ml-workspace` → **Storage Blob Data Contributor** on storage account
- [ ] Express Script principal → **Storage Blob Data Contributor** on storage (or containers)
- [ ] Datastore `claims_medicare_training` → container `medicare-training`
- [ ] Datastore `claims_reserve_training` → container `reserve-training`
- [ ] Test blob at `medicare-training/latest/claims.parquet` (optional)
- [ ] AML command job uses input + `TRAINING_DATA_URI` env var

---

## Portal steps (quick reference)

1. **Storage account** → Containers → Create `medicare-training`, `reserve-training`
2. **AML workspace** → Data → Datastores → New → Azure Blob Storage
   - Name: `claims_medicare_training`
   - Storage account + container: `medicare-training`
3. Repeat for `claims_reserve_training` / `reserve-training`
4. **Access control (IAM)** on storage account → Add role assignment
   - Role: Storage Blob Data Contributor
   - Member: `claims-ml-workspace` managed identity
5. IAM → Express Script service principal → Storage Blob Data Contributor

## One-time setup (CLI)

```bash
cd medicare_classifier
az login
chmod +x scripts/setup-training-datastore.sh
./scripts/setup-training-datastore.sh
```

This script:

1. Creates blob containers `medicare-training` and `reserve-training`
2. Grants the AML workspace managed identity **Storage Blob Data Contributor** on the storage account
3. Registers AML datastores via `scripts/register_training_datastore.py`

### Environment variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `RESOURCE_GROUP` | `rg-claims-intelligence` | Resource group |
| `STORAGE_ACCOUNT` | `claimsmlstorage4cd105406` | Blob account for training data |
| `CREATE_STORAGE_ACCOUNT` | `false` | Set `true` to create account if missing |
| `WORKSPACE_NAME` | `claims-ml-workspace` | AML workspace |

## Express Script handoff

Write parquet to these paths after each weekly extract:

```
https://<storage>.blob.core.windows.net/medicare-training/latest/claims.parquet
https://<storage>.blob.core.windows.net/reserve-training/latest/claims.parquet
```

Or partitioned layout:

```
medicare-training/latest/year=2026/month=06/part-00000.parquet
```

**Column contract:** Same schema as `data/claims_data.csv` (Oracle view columns). Medicare target: `is_v1m_extracted` or `is_medicare_reportable`. Reserve target: `reserve_6`.

Grant Express Script's service principal **Storage Blob Data Contributor** on the storage account (or container scope).

## AML command job — mount training data

### Option A: Datastore path input (recommended)

```python
from azure.ai.ml import command, Input
from azure.ai.ml.constants import AssetTypes

job = command(
    code="./",
    command="python train_medicare_classifier.py",
    environment="azureml:medicare-training-env:1",
    compute="claims-compute",
    inputs={
        "training_data": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml://datastores/claims_medicare_training/paths/latest/",
        ),
    },
    environment_variables={
        "TRAINING_DATA_URI": "${{inputs.training_data}}",
    },
)
```

### Option B: Direct URI (wasbs)

```bash
export TRAINING_DATA_URI="wasbs://medicare-training@claimsmlstorage4cd105406.blob.core.windows.net/latest/claims.parquet"
python train_medicare_classifier.py
```

## Training script

`train_medicare_classifier.py` resolves data in this order:

1. `TRAINING_DATA_URI` — mounted folder, single `.parquet`/`.csv`, or `wasbs://` path
2. `TRAINING_DATA_PATH` — local file override
3. `data/claims_data.csv` — default for local dev

```bash
# Local
python train_medicare_classifier.py

# Point at parquet
TRAINING_DATA_URI=./data/claims.parquet python train_medicare_classifier.py
```

## One-time setup (CLI)

```bash
# List containers
az storage container list --account-name claimsmlstorage4cd105406 --auth-mode login -o table

# List AML datastores
az ml datastore list --workspace-name claims-ml-workspace -g rg-claims-intelligence -o table

# Upload test file (dev)
az storage blob upload \
  --account-name claimsmlstorage4cd105406 \
  --container-name medicare-training \
  --name latest/claims.parquet \
  --file data/claims.parquet \
  --auth-mode login
```

## Reserve forecaster

Use the same storage account and `claims_reserve_training` datastore. When `train_reserve_forecaster.py` is added to `claims-intelligence`, set:

```bash
TRAINING_DATA_URI="${{inputs.training_data}}"
```

with input path `azureml://datastores/claims_reserve_training/paths/latest/`.
