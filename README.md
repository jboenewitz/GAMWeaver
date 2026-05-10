# Bike Rental Prediction Web Application

A full-stack web application for bike rental prediction using IGANN (Interpretable Generalized Additive Neural Networks).

## Project Structure

```
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py         # FastAPI application
│   │   ├── models.py       # Pydantic models
│   │   ├── ml_service.py   # ML service with IGANN
│   │   └── data_processing.py  # Data preprocessing
│   └── requirements.txt
├── frontend/               # React frontend (Vite)
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── api/          # API service
│   │   ├── App.jsx       # Main app
│   │   └── main.jsx      # Entry point
│   ├── package.json
│   └── vite.config.js
├── bike.csv               # Dataset
└── README.md
```

## Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

## Quick Start

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

- API Documentation: `http://localhost:8000/docs`

### 2. Frontend Setup

Open a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. **Load Data**: Click "Load Dataset" to load the bike rental data
2. **Train Model**: Configure the number of estimators and click "Train IGANN Model"
3. **View Results**: Once trained, you'll see:
   - Model metrics (RMSE, MAE)
   - Shape functions showing feature effects
   - Predictions vs Actual scatter plot
   - Hourly rental patterns
4. **Make Predictions**: Use the prediction form to predict bike rentals for specific conditions

## API Endpoints

| Endpoint                           | Method | Description                |
| ---------------------------------- | ------ | -------------------------- |
| `/api/health`                      | GET    | Health check               |
| `/api/data/load`                   | POST   | Load dataset               |
| `/api/data/summary`                | GET    | Get data summary           |
| `/api/data/hourly-pattern`         | GET    | Get hourly rental patterns |
| `/api/model/train`                 | POST   | Train IGANN model          |
| `/api/model/metrics`               | GET    | Get model metrics          |
| `/api/model/shape-functions`       | GET    | Get shape function data    |
| `/api/model/predictions-vs-actual` | GET    | Get predictions vs actual  |
| `/api/predict`                     | POST   | Make single prediction     |
| `/api/predict/batch`               | POST   | Make batch predictions     |

## Features

- **Interactive Training**: Configure and train IGANN models with customizable parameters
- **Shape Function Visualization**: See how each feature affects predictions
- **Real-time Predictions**: Make predictions with custom input values
- **Data Exploration**: View hourly patterns and data distributions
- **Model Evaluation**: Track RMSE and MAE metrics

## Technology Stack

### Backend

- FastAPI - Modern Python web framework
- IGANN - Interpretable Generalized Additive Neural Networks
- scikit-learn - Machine learning utilities
- pandas - Data manipulation

### Frontend

- React 18 - UI framework
- Vite - Build tool
- Plotly.js - Interactive charts
- Tailwind CSS - Styling
- Axios - HTTP client

## Deployment

For production deployment:

### Backend

```bash
cd backend
pip install gunicorn
gunicorn app.main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

Use a single backend worker unless model/data state is externalized, because the
current ML runtime state is kept in-process.

### Frontend

```bash
cd frontend
npm run build
# Serve the dist/ folder with any static file server
```

### Public Demo Deployment

For the Azure Static Web Apps + Docker-on-App-Service + PostgreSQL deployment flow,
see [DEPLOY.md](DEPLOY.md).

## License

MIT License
