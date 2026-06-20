# Azure Deploy — Medicare Classifier Streamlit UI

Deploy the **Streamlit** front-end as a second Container App. It calls the Medicare Classifier API via `API_URL`.

**Prerequisite:** `medicare-classifier-api` deployed and healthy.

| Component | Image | Port |
|-----------|--------|------|
| API | `ghcr.io/deepakguptaaiml/medicare-classifier:latest` | 8000 |
| **Streamlit UI** | `ghcr.io/deepakguptaaiml/medicare-classifier-streamlit:latest` | 8501 |

---

## Step 0 — Push code & wait for CI

After pushing to GitHub, confirm CI is green. Make the **streamlit** package public:

GitHub → **Packages** → `medicare-classifier-streamlit` → **Public**

---

## Step 1 — Get your API URL

Azure Portal → **`medicare-classifier-api`** → **Overview** → **Application Url**

Example: `https://medicare-classifier-api.xxxx.westus2.azurecontainerapps.io`

Copy this for `API_URL` (no trailing slash).

---

## Step 2 — Create Streamlit Container App (Portal)

1. **Create a resource** → **Container Apps**
2. **Basics**
   - Resource group: `rg-claims-intelligence`
   - Container app name: **`medicare-classifier-ui`**
   - Container Apps environment: **Use existing** → `claims-env`
3. **Container**
   - Image: `deepakguptaaiml/medicare-classifier-streamlit`
   - Tag: `latest`
4. **Environment variables**

   | Name | Value |
   |------|--------|
   | `API_URL` | `https://<medicare-classifier-api-fqdn>` |

5. **Ingress**
   - Target port: **8501**
6. **Create**

---

## Step 3 — Use the UI

Open **`medicare-classifier-ui`** Application Url. Sidebar should show **Connected to API**.

---

## Local Streamlit against Azure API

```bash
API_URL=https://<your-api-fqdn> streamlit run streamlit_app.py
```

---

## Architecture

```
User browser → medicare-classifier-ui (:8501)
                    ↓ HTTP (API_URL)
              medicare-classifier-api (:8000)
                    ↓
              medicare_classifier.pkl
```
