# Deployment Guide (GitHub Pages + Azure App Service)

This guide describes a public demo deployment for:

- Frontend: GitHub Pages
- Backend: Azure App Service for Linux
- Database: PostgreSQL (production), SQLite fallback for local dev

## Required GitHub Secrets / Variables

### GitHub Actions Secrets

- `AZURE_WEBAPP_NAME` = `<AZURE_WEBAPP_NAME>`
- `AZURE_WEBAPP_PUBLISH_PROFILE` = Azure publish profile XML (from the App Service)

### Optional GitHub Actions Variables

- `VITE_API_BASE_URL` (recommended)
  - Example: `https://<AZURE_WEBAPP_NAME>.azurewebsites.net/api`
- `VITE_PUBLIC_APP_URL` (recommended)
  - Example: `https://<GITHUB_USERNAME>.github.io/<REPO_NAME>/`

## Azure App Service Configuration

### App Settings (Environment Variables)

Set these in the Azure Web App configuration:

- `ENVIRONMENT=production`
- `DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<db>?sslmode=require`
- `CORS_ALLOW_ORIGINS=https://<GITHUB_USERNAME>.github.io`
- `ALLOW_DESTRUCTIVE_ACTIONS=false`
- `DEMO_ADMIN_SECRET=<strong-random-secret>` (optional, if you want protected admin actions)
- `SUPERADMIN_USERNAME=superadmin`
- `SUPERADMIN_PASSWORD=<strong-random-password>`
- `AUTH_TOKEN_SECRET=<strong-random-secret>`
- `ADMIN_TOKEN_TTL_HOURS=8` (optional)
- `INVITE_TOKEN_TTL_HOURS=168` (optional)

### Startup Command

Use this startup command (from the repo root):

```
./backend/startup.sh
```

## GitHub Pages Configuration

1. In GitHub: Repository Settings -> Pages
2. Set Source to **GitHub Actions**
3. The workflow deploys the Vite `dist/` output automatically

## Environment Files (Local Development)

### Backend

Copy and adjust:

```
backend/.env.example -> backend/.env
```

### Frontend

Copy and adjust:

```
frontend/.env.example -> frontend/.env
```

## GitHub Actions Workflows

- `.github/workflows/deploy-frontend.yml`
- `.github/workflows/deploy-backend.yml`

## Local Testing

### Backend

```
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```
cd frontend
npm install
npm run dev
```

## Manual Steps Still Required

- Create the Azure App Service (Linux) and PostgreSQL database
- Set Azure App Service app settings (env vars above)
- Ensure the startup script is executable (`chmod +x backend/startup.sh`)
- Download and add the publish profile to GitHub secrets
- Set GitHub Pages to use GitHub Actions
- Update GitHub Variables if you want to configure `VITE_API_BASE_URL`

## First Deploy Order

1. Create Azure resources (App Service + PostgreSQL)
2. Configure App Service settings and startup command
3. Add GitHub secrets for Azure
4. Push to `main` (workflows will deploy automatically)
5. Verify backend health: `https://<AZURE_WEBAPP_NAME>.azurewebsites.net/health`
6. Verify frontend: `https://<GITHUB_USERNAME>.github.io/<REPO_NAME>/`
