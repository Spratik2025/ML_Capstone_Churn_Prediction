FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app ./app

# Copy trained ML model and metadata
COPY models ./models

# Copy training evaluation metrics
COPY reports ./reports

# Create monitoring directory
RUN mkdir -p monitoring

# Copy reference data for drift monitoring
COPY monitoring/reference_data.csv ./monitoring/reference_data.csv

# Verify that the trained model exists
RUN test -f /app/models/model.joblib

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
