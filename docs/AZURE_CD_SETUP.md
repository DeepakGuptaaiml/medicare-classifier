# Azure CD Setup — Medicare Classifier

GitHub Actions **CI** builds and pushes Docker images to GHCR. **CD** (`.github/workflows/deploy-azure.yml`) runs after green CI on `main` and updates:

| App | Name |
|-----|------|
| API | `medicare-classifier-api` |
| Streamlit | `medicare-classifier-ui` |
| Resource group | `rg-claims-intelligence` |

```
git push main → CI (test, build, push GHCR) → Deploy to Azure (az containerapp update)
```

---

## Reuse claims-intelligence credentials

If you already set up CD for **claims-intelligence**, you can **reuse the same GitHub secret**:

| Secret | Notes |
|--------|--------|
| `AZURE_CREDENTIALS` | Same service principal with **Contributor** on `rg-claims-intelligence` |

Add this secret to the **medicare-classifier** GitHub repo (same JSON value).

Full Portal/CLI setup details: see `claims-intelligence/docs/AZURE_CD_SETUP.md` (same steps, same resource group).

---

## One-time bootstrap

CD **updates** existing Container Apps; it does not create them. Bootstrap once:

1. [AZURE_API_DEPLOY.md](AZURE_API_DEPLOY.md) — create `medicare-classifier-api`
2. [AZURE_STREAMLIT_DEPLOY.md](AZURE_STREAMLIT_DEPLOY.md) — create `medicare-classifier-ui` with `API_URL`

---

## GHCR packages public

| Package | GitHub path |
|---------|-------------|
| API | **Packages** → `medicare-classifier` → **Public** |
| Streamlit | **Packages** → `medicare-classifier-streamlit` → **Public** |

Or add `GHCR_TOKEN` + `GHCR_USER` secrets (same as claims-intelligence).

### Hugging Face token (RAG agent)

Add GitHub secret **`HF_API_TOKEN`** with a [Hugging Face access token](https://huggingface.co/settings/tokens) (Inference API scope).

CD workflow automatically:
1. Sets `hf-api-token` secret on **`medicare-classifier-api`**
2. Maps env var `HF_API_TOKEN=secretref:hf-api-token`

For manual Portal setup: **medicare-classifier-api** → **Containers** → add env var `HF_API_TOKEN`.

RAG endpoints:
- `GET /rag/status` — agent readiness
- `POST /ask` — policy Q&A

---

## What happens on each push to `main`

1. **CI** runs tests, builds API + Streamlit images, pushes tags `latest` and `{7-char-sha}`.
2. **Deploy to Azure** triggers on CI success.
3. `az containerapp update` sets each app to the SHA-tagged image.
4. Workflow curls `/health` and `/model/sample` on the live API.

Streamlit `API_URL` is **unchanged** across image updates.

---

## Troubleshooting

See `claims-intelligence/docs/AZURE_CD_SETUP.md` for:

- `AADSTS7000215` (wrong client secret Value vs Secret ID)
- `No subscriptions found` (wrong subscriptionId or missing Contributor)
- `MANIFEST_UNKNOWN` (missing SHA tag — push a new commit to trigger CI)

For medicare-classifier specifically, confirm app names in `deploy-azure.yml` match Portal:

- `medicare-classifier-api`
- `medicare-classifier-ui`

---

## Troubleshooting — `SyntaxError: Unexpected non-whitespace character after JSON`

**Symptom (Deploy to Azure → Log in to Azure):**
```
Login failed with SyntaxError: Unexpected non-whitespace character after JSON at position 10
Double check if the 'auth-type' is correct.
```

**Cause:** The **`AZURE_CREDENTIALS`** secret value is **not valid JSON**. The `azure/login@v2` action expects a single JSON object — nothing before `{`, nothing after `}`.

**Common copy/paste mistakes:**

| Mistake | Example |
|---------|---------|
| Label prefix in the value | `AZURE_CREDENTIALS={"clientId": ...}` |
| Markdown / backticks | ` ```json {"clientId": ...} ``` ` |
| Smart quotes from Word/Notes | `"clientId"` (curly quotes) |
| Trailing comma on last field | `"tenantId": "...",}` |
| Only the client secret pasted | `abc~8Q~longString...` (not JSON) |
| Secret ID instead of full JSON | Single GUID only |
| Two JSON blobs concatenated | `{...}{...}` |

**Fix:**

1. GitHub → **medicare-classifier** → **Settings** → **Secrets and variables** → **Actions**
2. **Edit** `AZURE_CREDENTIALS` (or delete and recreate)
3. Paste **only** this shape (replace with your real values from Entra ID + Subscriptions):

```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "abc~8Q~pasteValueColumnFromPortal",
  "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
}
```

**Rules:**
- Double quotes `"` only (straight quotes, not curly)
- No trailing comma after `tenantId`
- `clientSecret` = **Value** column from Portal (not Secret ID)
- `subscriptionId` = Subscription **GUID** from Portal → Subscriptions
- No extra spaces or text outside the `{ ... }`

4. **Actions** → **Deploy to Azure** → **Re-run all jobs**

**If you no longer have the original JSON:** rebuild from Portal (App registration → Overview + Certificates & secrets → Subscriptions) or run in **Azure Cloud Shell**:

```bash
az login
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
az ad sp list --display-name "github-claims-intelligence-cd" --query "[0].appId" -o tsv
# Create NEW client secret in Portal if needed, then assemble JSON manually
```

**Validate locally** (Mac Terminal — paste JSON into a file, do not commit):

```bash
python3 -c 'import json; json.load(open("azure-creds.json"))' && echo "JSON OK"
```

**Note:** `auth-type` is correct in our workflow (`creds:` JSON is the default). The error is almost always malformed secret content, not the workflow YAML.

