FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy model files
COPY ai-models /app/models

# Install Python dependencies
COPY server/ai_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY server/ai_api /app

# Expose API port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "model_serving:app", "--host", "0.0.0.0", "--port", "8000"]