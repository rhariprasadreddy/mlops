# Use a slim Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from current directory to container's /app
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the port your app will run on
EXPOSE 8001

# Run the app using Gunicorn
# Ensure 'app:app' matches:
# - app.py (your Python file)
# - 'app' object inside app.py (e.g., app = Flask(__name__) or app = FastAPI())
CMD ["gunicorn", "--bind", "0.0.0.0:8001", "app:app"]


