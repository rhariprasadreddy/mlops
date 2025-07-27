# syntax=docker/dockerfile:1
FROM python:3.12-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies (add only what's necessary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files for installation
COPY docker-requirements.txt ./requirements.txt

# Upgrade pip and install dependencies from requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application code
COPY app.py ./
COPY templates ./templates

# Set environment variables for scalability
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
    MODEL_NAME=FashionMNIST-BestModel \
    MODEL_STAGE=Production

# Expose the application port (FastAPI/Flask)
EXPOSE 5001

# Set entry point using gunicorn for production (can be overridden)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]