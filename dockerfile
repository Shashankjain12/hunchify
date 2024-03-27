# Use Python 3.10.0 as base image
FROM python:3.10.0

# Set working directory
WORKDIR /app

# Copy requirements.txt to the container
COPY . .

# Install Prefect and other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 4200
EXPOSE 4200

# Define the command to start the Prefect server
CMD prefect server start &

# Start the Prefect worker with a managed pool named 'my-managed-pool'
CMD prefect worker start --pool my-managed-pool &

# Run the Python script create_deployment.py
CMD python prefect-dags/create_deployment.py
