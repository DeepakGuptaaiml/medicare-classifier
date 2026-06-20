# Medicare Classifier

Medicare Claims Intelligence Platform — **Model 1: Medicare Classifier**

Predicts whether a claim is **Medicare reportable** under MMSEA Section 111 (V1M extraction), using workers' comp and liability claim intake features.

> **Python version:** Use **3.12** for notebook + SHAP. See [Python 3.12 + SHAP](#python-312--full-shap) below. Phase 2 Docker/CI will pin `3.12` (same as `claims-intelligence`).

## Phase 1 (current)

| Artifact | Description |
|----------|-------------|
| `Medicare_Classifier_Notebook.ipynb` | AML-style training notebook (58 cells) |
| `Medicare_Classifier_Notebook_executed.ipynb` | Executed notebook with outputs |
| `models/medicare_classifier.pkl` | Best tuned model + metadata |
| `models/preprocess_config.json` | Encoding / imputation config for API |
| `train_medicare_classifier.py` | Standalone training script (same pipeline) |

## Target

- **`is_medicare_reportable`** — renamed from `is_v1m_extracted` in `claims_data.csv`
- **Primary metric:** Recall (missing a reportable claim is costlier than a false positive)

## Features (20)

Raw: `data_set`, `pay_cat`, `pay_code`, `pay_type`, `paid_1`, `paid_3`, `amount`, `proc_unit`, `cont_num`, `claim_open`

Engineered: `date_v1m_xmit_flag`, `is_us_claimant`, `orm_threshold_met`, `tpoc_threshold_met`, `is_wc`, `pay_code_bucket`, `is_excluded_coverage`, `is_excluded_line`, `days_open`, `age_at_event`

## Quick start

```bash
cd medicare_classifier
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
python train_medicare_classifier.py
```

Regenerate and execute notebook (with SHAP):

```bash
python generate_medicare_notebook.py
jupyter nbconvert --to notebook --execute Medicare_Classifier_Notebook.ipynb \
  --output Medicare_Classifier_Notebook_executed.ipynb \
  --ExecutePreprocessor.timeout=900
```

## Python 3.12 + full SHAP

**Why:** SHAP depends on `numba`, which does not reliably support **Python 3.14** on Mac. On 3.14 the notebook falls back to sklearn feature-importance plots instead of SHAP beeswarm/waterfall charts.

**Fix — dedicated 3.12 venv:**

```bash
cd medicare_classifier
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt   # includes shap>=0.44

# Re-run SHAP cells only, or re-execute full notebook:
jupyter nbconvert --to notebook --execute Medicare_Classifier_Notebook.ipynb \
  --output Medicare_Classifier_Notebook_executed.ipynb \
  --ExecutePreprocessor.timeout=900
```

**No retrain needed** — `models/medicare_classifier.pkl` is unchanged; only notebook outputs (SHAP plots) refresh.

| Environment | SHAP |
|-------------|------|
| Python 3.14 (Mac default) | Fallback: feature importance |
| **Python 3.12** (recommended) | Full SHAP summary + local explanations |
| Docker `python:3.12-slim` (Phase 2) | Full SHAP in CI/production |

Install Python 3.12 on Mac if needed: `brew install python@3.12`

## Phase 2 — MLOps (FastAPI → Azure)

| Artifact | Description |
|----------|-------------|
| `app/main.py` | FastAPI — `/health`, `/predict`, `/model/info`, `/model/options`, `/model/sample` |
| `streamlit_app.py` | Streamlit UI calling the API |
| `tests/test_api.py` | Pytest suite (5 tests) |
| `Dockerfile` / `Dockerfile.streamlit` | Python 3.12 containers |
| `.github/workflows/ci.yml` | Test + build + push to GHCR |
| `.github/workflows/deploy-azure.yml` | CD to `medicare-classifier-api` / `medicare-classifier-ui` |
| `docs/AZURE_*.md` | Azure bootstrap + CD setup |

### Run locally

```bash
cd medicare_classifier
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-api.txt
uvicorn app.main:app --reload --port 8000
```

Second terminal:

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

Tests:

```bash
pip install pytest httpx
pytest tests/ -v
```

### Deploy to Azure

1. Push repo to GitHub (new repo recommended, e.g. `medicare-classifier`)
2. Make GHCR packages **Public**
3. Bootstrap Container Apps — see [docs/AZURE_API_DEPLOY.md](docs/AZURE_API_DEPLOY.md) and [docs/AZURE_STREAMLIT_DEPLOY.md](docs/AZURE_STREAMLIT_DEPLOY.md)
4. Add `AZURE_CREDENTIALS` secret (reuse from claims-intelligence) — [docs/AZURE_CD_SETUP.md](docs/AZURE_CD_SETUP.md)
5. Push to `main` → CI builds → CD deploys SHA-tagged images

**Azure app names:** `medicare-classifier-api`, `medicare-classifier-ui` (same RG: `rg-claims-intelligence`)

### SHAP (deferred)

SHAP explainability is Phase 1 notebook only for now. Docker/CI use Python 3.12 so SHAP can be added later without changing the deployment stack.
