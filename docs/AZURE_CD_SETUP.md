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
