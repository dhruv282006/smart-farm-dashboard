# Smart Farm Dashboard

A demo Smart Farm dashboard with weather, area-aware agricultural insights, and a small ML pipeline for yield prediction. This project is intended as a portfolio piece for ML internships and demonstrates:

- React frontend with Leaflet/OpenStreetMap map overlays
- Express backend that proxies weather APIs, synthesizes area suggestions, and computes agriculture insights
- Small scikit-learn model training pipeline and a CLI prediction wrapper

This repository is a demo: replace synthetic datasets with historical, labeled agricultural data for production-ready models.

## Quickstart

Prerequisites:
- Node.js 18+ (or latest LTS)
- Python 3.10+ and pip

Backend
```powershell
cd backend
# (optional) create and activate a Python venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm install
# train debug/demo model
npm run train-model
# start backend
npm start
```

Frontend
```powershell
cd frontend
npm install
npm start
```

Open http://localhost:3000 in your browser (CRA will proxy API calls to backend on port 5000).

## Production notes
- Replace the demo ML dataset in `backend/ml/train_model.py` with historical features and labels.
- Use a persistent datastore (S3 or artifact storage) for trained models and version metadata.
- Add unit/integration tests and a CI pipeline (GitHub Actions) to run tests and build the frontend.

## Files of interest
- `backend/server.js` — main API server
- `frontend/src/components/MapView.js` — Leaflet integration
- `backend/ml/train_model.py` — training script that outputs model and metadata
- `backend/ml/predict.py` — CLI wrapper for predictions

## License
MIT
