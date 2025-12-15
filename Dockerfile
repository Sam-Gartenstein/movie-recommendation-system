# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts into the image
COPY src ./src
COPY app ./app
COPY artifacts ./artifacts

# Make src importable as a package
ENV PYTHONPATH=/app/src

# Expose the port Gradio will listen on
EXPOSE 7860

# Start the Gradio app when the container runs
CMD ["python", "-m", "app.gradio_app"]

