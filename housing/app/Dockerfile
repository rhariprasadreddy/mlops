# Use official Python image
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app.py .

# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Expose port
EXPOSE 8000

# Run app
CMD ["python", "app.py"]