# CardioPulse AI - Docker Deployment Configuration
FROM python:3.10-slim

# Install system audio and build libraries for librosa & scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install dependencies & production WSGI server
RUN pip install --no-cache-dir -r requirements.txt gunicorn waitress

# Copy application files
COPY . .

# Set environment variables
ENV PORT=5000
EXPOSE 5000

# Start server using production WSGI runner (dynamically binds to Render's $PORT)
CMD ["sh", "-c", "python -m waitress --host=0.0.0.0 --port=${PORT:-5000} app:app"]
