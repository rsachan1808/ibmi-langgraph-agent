# Base image — Python 3.11 slim keeps the container small
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first — Docker caches this layer
# If requirements don't change, this layer is not rebuilt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run when container starts
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]