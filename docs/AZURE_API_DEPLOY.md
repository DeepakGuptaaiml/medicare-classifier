# Azure Deploy — Medicare Classifier API (Container Apps)

Deploy the **FastAPI** Medicare Classifier to Azure Container Apps using the Docker image built by GitHub Actions.

**Image:** `ghcr.io/deepakguptaaiml/medicare-classifier:latest`  
**Port:** `8000`  
**Health:** `GET /health`  
**Streamlit UI:** deploy separately — see [AZURE_STREAMLIT_DEPLOY.md](AZURE_STREAMLIT_DEPLOY.md)

**Auto-deploy (CD):** after bootstrap, configure [AZURE_CD_SETUP.md](AZURE_CD_SETUP.md) so every green CI push updates Container Apps automatically.

---

## Prerequisites

- Azure subscription and CLI (`brew install azure-cli`)
- GitHub Actions **green** (image published to GHCR)
- GHCR package **Public**:
  - GitHub → repo → **Packages** → `medicare-classifier` → **Package settings** → **Public**
- Reuse existing resource group **`rg-claims-intelligence`** and environment **`claims-env`** from claims-intelligence (recommended)

---

## Option A — Azure Portal

1. **Create a resource** → **Container Apps**
2. **Basics**
   - Resource group: `rg-claims-intelligence`
   - Container app name: **`medicare-classifier-api`**
   - Region: same as claims-intelligence (e.g. West US 2)
3. **Container Apps environment** → **Use existing** → `claims-env`
4. **Container**
   - Registry: `ghcr.io`
   - Image: `deepakguptaaiml/medicare-classifier`
   - Tag: `latest`
   - CPU / Memory: 0.5 CPU, 1 Gi
5. **Ingress**
   - Enabled: **Yes**
   - Target port: **8000**
6. **Health probes**
   - HTTP GET `/health` on port **8000**
7. **Create**

### Verify

```bash
curl https://<your-app-fqdn>/health
curl https://<your-app-fqdn>/model/info
```

Browser: `https://<your-app-fqdn>/docs`

---

## Option B — Azure CLI

```bash
az login

export RESOURCE_GROUP=rg-claims-intelligence
export ENV_NAME=claims-env
export APP_NAME=medicare-classifier-api
export IMAGE=ghcr.io/deepakguptaaiml/medicare-classifier:latest

chmod +x scripts/deploy-azure-api.sh
./scripts/deploy-azure-api.sh
```

---

## Test the live API

```bash
curl -X POST https://<FQDN>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data_set": "WC",
    "pay_cat": "PI",
    "pay_code": 328,
    "pay_type": "SYS",
    "paid_1": 24488.39,
    "paid_3": 47490.06,
    "amount": 5019.64,
    "proc_unit": 49,
    "cont_num": 5414,
    "claim_open": 1,
    "date_v1m_xmit_flag": 0,
    "is_us_claimant": 1,
    "orm_threshold_met": 1,
    "tpoc_threshold_met": 1,
    "is_wc": 1,
    "pay_code_bucket": "ORM",
    "is_excluded_coverage": 0,
    "is_excluded_line": 0,
    "days_open": 0.0,
    "age_at_event": 70.5
  }'
```

---

## Architecture on Azure

```
Internet → medicare-classifier-api (:8000) → medicare_classifier.pkl
                ↓
         GET /health, POST /predict, GET /docs
```
