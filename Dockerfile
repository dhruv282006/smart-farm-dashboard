FROM node:18-alpine AS builder
WORKDIR /app

# Build frontend
COPY frontend/package.json frontend/package-lock.json* ./frontend/
COPY frontend/ ./frontend/
WORKDIR /app/frontend
RUN npm install --production=false --no-audit --no-fund
RUN npm run build

FROM node:18-alpine
WORKDIR /app
COPY backend/package.json ./backend/
COPY backend/server.js ./backend/
COPY --from=builder /app/frontend/build ./frontend/build
WORKDIR /app/backend
RUN npm install --production
EXPOSE 5000
CMD ["node", "server.js"]
