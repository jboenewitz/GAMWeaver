# Deployment Guide (Frontend + Backend on Azure App Service Code Deployments)

This guide deploys:

- Frontend: Azure App Service (Linux, Node.js)
- Backend: Azure App Service (Linux, Python/FastAPI)
- Database: Azure Database for PostgreSQL Flexible Server

## Backend Runtime

Backend startup command is defined in:

- `backend/startup.sh`

Build locally:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
```

Run locally:

```bash
ENVIRONMENT=development \
DATABASE_URL=sqlite:///$(pwd)/backend/igann_app.db \
CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:5173 \
ALLOW_DESTRUCTIVE_ACTIONS=true \
SUPERADMIN_USERNAME=superadmin \
SUPERADMIN_PASSWORD=change-me \
AUTH_TOKEN_SECRET=change-me \
bash backend/startup.sh
```

## Monorepo Build Helpers

- Root `requirements.txt` forwards to `backend/requirements.txt` so Azure App Service build automation can install Python dependencies from the repo root.
- Frontend App Service deploys only the `frontend/` folder.

## Required GitHub Secrets

Set these in GitHub (`Settings -> Secrets and variables -> Actions`):

- `AZURE_BACKEND_WEBAPP_NAME`
- `AZURE_BACKEND_WEBAPP_PUBLISH_PROFILE`
- `AZURE_FRONTEND_WEBAPP_NAME`
- `AZURE_FRONTEND_WEBAPP_PUBLISH_PROFILE`

## GitHub Actions Workflows

- Backend deploy: `.github/workflows/deploy-backend.yml`
  - Trigger: `push` to `main` and `workflow_dispatch`
  - Flow: upload repo to Azure App Service -> App Service installs Python dependencies -> run `bash backend/startup.sh`
- Frontend deploy: `.github/workflows/deploy-frontend.yml`
  - Trigger: `push` to `main` and `workflow_dispatch`
  - Deploy target: Azure App Service
  - Build env:
    - `VITE_API_BASE_URL=https://<AZURE_BACKEND_WEBAPP_NAME>.azurewebsites.net/api`
    - `VITE_BASE_PATH=/`
    - `VITE_ALLOW_DESTRUCTIVE_ACTIONS=false`

## Azure Runbook (First-Time Setup)

1. Keep spending limit enabled and create budget alerts at 70% and 90%.
2. Create Azure Database for PostgreSQL Flexible Server (`B1ms`) and create DB/user.
3. Create a backend Azure App Service:
   - OS: Linux
   - Publish: Code
   - Runtime stack: Python
4. In the backend App Service (`Environment variables`), set:
   - `ENVIRONMENT=production`
   - `DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<db>?sslmode=require`
   - `CORS_ALLOW_ORIGINS=https://<your-frontend-webapp-name>.azurewebsites.net`
   - `ALLOW_DESTRUCTIVE_ACTIONS=false`
   - `SUPERADMIN_USERNAME=superadmin`
   - `SUPERADMIN_PASSWORD=<strong-random-password>`
   - `AUTH_TOKEN_SECRET=<strong-random-secret>`
   - `DEMO_ADMIN_SECRET=<strong-random-secret>` (optional)
   - `ADMIN_TOKEN_TTL_HOURS=8` (optional)
   - `INVITE_TOKEN_TTL_HOURS=168` (optional)
   - `SCM_DO_BUILD_DURING_DEPLOYMENT=true`
5. In the backend App Service (`Configuration -> General settings`), set startup command to `bash backend/startup.sh`.
6. Download the backend App Service publish profile and store it in `AZURE_BACKEND_WEBAPP_PUBLISH_PROFILE`.
7. Set `AZURE_BACKEND_WEBAPP_NAME` GitHub secret.
8. Create a second Azure App Service for the frontend:
    - OS: Linux
    - Runtime stack: Node 22 LTS
    - Publish: Code
9. Download the frontend App Service publish profile and store it in `AZURE_FRONTEND_WEBAPP_PUBLISH_PROFILE`.
10. Set `AZURE_FRONTEND_WEBAPP_NAME` GitHub secret.
11. In the frontend App Service (`Configuration -> General settings`), set startup command to `node server.js`.
12. If you use Azure Portal Deployment Center instead of these repo workflows, disable or remove any duplicate workflow so only one deployment pipeline owns each app.
13. Push to `main` (or run workflows manually): backend first, frontend second.
14. After frontend URL exists, update backend `CORS_ALLOW_ORIGINS` to the exact frontend App Service URL and restart backend once.

## Test Plan (Post-Deploy)

- `GET https://<AZURE_BACKEND_WEBAPP_NAME>.azurewebsites.net/health`
- `POST https://<AZURE_BACKEND_WEBAPP_NAME>.azurewebsites.net/api/data/load`
- Verify user/auth data survives backend restart (PostgreSQL persistence).
- Open frontend App Service URL and confirm training/prediction works without CORS errors.
- Confirm destructive actions are blocked without superadmin access.

## Notes

- The backend deployment includes `bike.csv` from the repo root.
- If dataset location is custom, set `DATA_FILE_PATH` in App Service environment variables.
