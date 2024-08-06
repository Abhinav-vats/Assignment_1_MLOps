# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5005 for the Flask app and port 5000 for MLflow
EXPOSE 5005
EXPOSE 5000

# Copy the dev_init.sh script
COPY src/dev_init.sh /app/dev_init.sh

# Make the dev_init.sh script executable
RUN chmod +x /app/dev_init.sh

# Run the dev_init.sh script and then train_model.py
CMD ["/bin/bash", "-c", "/app/dev_init.sh && sleep 10 && python /app/train_model.py"]
