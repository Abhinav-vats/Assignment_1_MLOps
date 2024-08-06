# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5005 for the Flask app and port 5000 for MLflow
EXPOSE 5005
EXPOSE 5000

# Run the train_model.py script
CMD ["python", "/app/src/train_model.py"]
