# Telco Churn API — production image
# Build:  docker build -t telco-churn-api .
# Run:    docker run -p 8000:8000 telco-churn-api

FROM python:3.11-slim

# Security: run as non-root
RUN useradd --create-home appuser
WORKDIR /app

# Install dependencies first so Docker layer caching skips reinstalls
# when only source code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what serving needs: source code and the trained model artifact
COPY src/ ./src/
COPY models/churn_model.joblib ./models/churn_model.joblib

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
