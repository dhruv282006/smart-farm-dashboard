Quick deploy options

Docker (recommended):
1. Build image:
   docker build -t smart-farm-dashboard:latest .
2. Run container:
   docker run -p 5000:5000 smart-farm-dashboard:latest

Heroku (quick):
1. Create heroku app and push repo (or use Heroku CLI to deploy). The `Procfile` runs the backend which will serve the frontend build if present.

Notes:
- Docker will build the frontend and copy the static build into the backend, so the app is served from the same process.
- Ensure OpenWeatherMap API key in `backend/server.js` is valid.
