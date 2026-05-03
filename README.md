# 🌴 Goa AI Recommender — MLOps Pipeline

> **ML-powered popularity ranking for Goa tourist spots**  
> FastAPI · Scikit-learn · MLflow · Docker · GitHub Actions CI/CD · EC2

---

## 📁 Project Structure

```
MlOps_Sem_End/
├── app/
│   ├── main.py          # FastAPI application
│   └── schema.py        # Pydantic schema
├── model/
│   └── rank_model.pkl   # Trained ML model
├── frontend/
│   └── index.html       # Web UI
├── tests/
│   └── test_api.py      # Pytest async test suite
├── .github/
│   └── workflows/
│       └── ci-cd.yml    # GitHub Actions pipeline
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # Full stack (API + MLflow + Nginx)
├── nginx.conf           # Reverse proxy config
├── requirements.txt     # Pinned dependencies
├── pyproject.toml       # Pytest & Ruff config
└── .dockerignore
```

---

## 🚀 Quick Start

### 1. Run locally (Python)
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Visit → http://localhost:8000/docs

### 2. Run with Docker
```bash
docker build -t goa-recommender .
docker run -p 8000:8000 goa-recommender
```

### 3. Run full stack with Docker Compose
```bash
docker compose up --build
```
| Service  | URL                        |
|----------|----------------------------|
| API      | http://localhost:8000      |
| Swagger  | http://localhost:8000/docs |
| Frontend | http://localhost:80        |
| MLflow   | http://localhost:5000      |

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

16 async tests covering: health, predict (happy path + error cases), info, root.

---

## 🐳 Docker Details

- **Multi-stage build** — builder + slim runtime image
- **Non-root user** — runs as `appuser` for security
- **Health check** — Docker polls `/health` every 30s
- **Two workers** — uvicorn starts with 2 workers

---

## ⚙️ CI/CD Pipeline (GitHub Actions)

```
Push to main
    │
    ▼
[Test & Lint]  ──→  pytest 16 tests + ruff lint
    │
    ▼
[Build & Push]  ──→  Docker image → GitHub Container Registry (ghcr.io)
    │
    ▼
[Deploy]  ──→  SSH into EC2, docker compose up --force-recreate
    │
    ▼
[Notify]  ──→  Print pipeline summary
```

### Required GitHub Secrets

| Secret        | Description                            |
|---------------|----------------------------------------|
| `EC2_HOST`    | Public IP of your EC2 instance         |
| `EC2_USER`    | SSH username (e.g., `ubuntu`)          |
| `EC2_SSH_KEY` | Private SSH key content (PEM format)   |

---

## 🌐 API Reference

| Method | Endpoint   | Description              |
|--------|------------|--------------------------|
| GET    | `/health`  | Liveness / readiness check |
| GET    | `/info`    | Service metadata          |
| GET    | `/docs`    | Swagger UI                |
| POST   | `/predict` | Run popularity prediction |

### Predict Request Body
```json
{
  "rating": 4.5,
  "review_count": 120,
  "category": "beach",
  "text": "sunset beach party vibes"
}
```

### Predict Response
```json
{
  "prediction": 4.2,
  "category": "beach",
  "rating": 4.5,
  "review_count": 120
}
```

---

## 🛠️ EC2 Deployment (First Time)

```bash
# On your EC2 instance
sudo apt update && sudo apt install -y docker.io docker-compose-plugin git
sudo systemctl start docker

# Clone repo
git clone https://github.com/<your-username>/MlOps_Sem_End.git ~/goa-recommender
cd ~/goa-recommender

# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u <username> --password-stdin

# Start
docker compose up -d
```

---

## 📊 MLflow Tracking

MLflow is included in the docker-compose stack and tracks:
- Model training runs
- Metrics and parameters
- Artifact storage

Access at → http://localhost:5000

---

*Built with ❤️ for Goa travel recommendations*
