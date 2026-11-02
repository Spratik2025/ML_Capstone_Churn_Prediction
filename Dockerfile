FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app ./app

# Copy trained model and metadata
COPY models ./models

# Copy evaluation metrics
COPY reports ./reports

# Create monitoring directory
RUN mkdir -p monitoring

# Copy reference data
COPY monitoring/reference_data.csv ./monitoring/reference_data.csv

# Verify model file exists
RUN test -f /app/models/model.joblib

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
