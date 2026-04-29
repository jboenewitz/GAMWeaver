# Deployment Guide (Docker Backend on Azure + Azure Static Web Apps Frontend)

This guide deploys:

- Frontend: Azure Static Web Apps
- Backend: Docker container on Azure App Service (Linux, Web App for Containers)
- Database: Azure Database for PostgreSQL Flexible Server

## Backend Docker Runtime

Backend container image is defined in:

- `backend/Dockerfile`

Build locally:

```bash
docker build -f backend/Dockerfile -t bike-backend:local .
```

Run locally:

```bash
docker run --rm -p 8000:8000 \
  -e ENVIRONMENT=development \
  -e DATABASE_URL=sqlite:////app/backend/igann_app.db \
  -e CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:5173 \
  -e ALLOW_DESTRUCTIVE_ACTIONS=true \
  -e SUPERADMIN_USERNAME=superadmin \
  -e SUPERADMIN_PASSWORD=change-me \
  -e AUTH_TOKEN_SECRET=change-me \
  bike-backend:local
```

## Required GitHub Secrets

Set these in GitHub (`Settings -> Secrets and variables -> Actions`):

- `AZURE_WEBAPP_NAME`
- `AZURE_WEBAPP_PUBLISH_PROFILE`
- `AZURE_STATIC_WEB_APPS_API_TOKEN`
- `AZURE_ACR_LOGIN_SERVER` (example: `myregistry.azurecr.io`)
- `AZURE_ACR_USERNAME`
- `AZURE_ACR_PASSWORD`

## GitHub Actions Workflows

- Backend deploy: `.github/workflows/deploy-backend.yml`
  - Trigger: `push` to `main` and `workflow_dispatch`
  - Flow: build Docker image -> push to Azure Container Registry -> deploy image to App Service
- Frontend deploy: `.github/workflows/deploy-frontend.yml`
  - Trigger: `push` to `main` and `workflow_dispatch`
  - Deploy target: Azure Static Web Apps
  - Build env:
    - `VITE_API_BASE_URL=https://<AZURE_WEBAPP_NAME>.azurewebsites.net/api`
    - `VITE_BASE_PATH=/`
    - `VITE_ALLOW_DESTRUCTIVE_ACTIONS=false`

## Azure Runbook (First-Time Setup)

1. Keep spending limit enabled and create budget alerts at 70% and 90%.
2. Create Azure Database for PostgreSQL Flexible Server (`B1ms`) and create DB/user.
3. Create Azure Container Registry (ACR).
4. Enable ACR admin user and copy:
   - Login server
   - Username
   - Password
5. Create Azure App Service (Linux, Web App for Containers).
6. In App Service (`Deployment Center -> Container Registry`), configure:
   - Source: Azure Container Registry
   - Registry: your ACR
   - Image: `bike-rental-backend`
   - Tag: `latest`
7. In App Service (`Environment variables`), set:
   - `ENVIRONMENT=production`
   - `DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<db>?sslmode=require`
   - `CORS_ALLOW_ORIGINS=https://<your-static-web-app-hostname>`
   - `ALLOW_DESTRUCTIVE_ACTIONS=false`
   - `SUPERADMIN_USERNAME=superadmin`
   - `SUPERADMIN_PASSWORD=<strong-random-password>`
   - `AUTH_TOKEN_SECRET=<strong-random-secret>`
   - `DEMO_ADMIN_SECRET=<strong-random-secret>` (optional)
   - `ADMIN_TOKEN_TTL_HOURS=8` (optional)
   - `INVITE_TOKEN_TTL_HOURS=168` (optional)
   - `WEBSITES_PORT=8000`
8. Download App Service publish profile and store it in `AZURE_WEBAPP_PUBLISH_PROFILE`.
9. Set `AZURE_WEBAPP_NAME` GitHub secret.
10. Create Azure Static Web App (Free plan), connect repo `main`, set app location `frontend`, output location `dist`.
11. Save Static Web Apps deployment token as `AZURE_STATIC_WEB_APPS_API_TOKEN`.
12. Push to `main` (or run workflows manually): backend first, frontend second.
13. After frontend URL exists, update backend `CORS_ALLOW_ORIGINS` to exact SWA URL and restart backend once.

## Test Plan (Post-Deploy)

- `GET https://<AZURE_WEBAPP_NAME>.azurewebsites.net/health`
- `POST https://<AZURE_WEBAPP_NAME>.azurewebsites.net/api/data/load`
- Verify user/auth data survives backend restart (PostgreSQL persistence).
- Open Static Web App URL and confirm training/prediction works without CORS errors.
- Confirm destructive actions are blocked without superadmin access.

## Notes

- Backend Docker image includes `bike.csv` and startup script.
- If dataset location is custom, set `DATA_FILE_PATH` in App Service environment variables.
