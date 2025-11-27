# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, API, and precomputed artifacts into the image
COPY src ./src
COPY app ./app
COPY artifacts ./artifacts

# Make src importable as a package
ENV PYTHONPATH=/app/src

# Expose the port FastAPI will listen on
EXPOSE 8000

# Start the FastAPI server when the container runs
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
