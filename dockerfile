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

# Make the script executable
RUN chmod +x /app/startup.sh

# Define the command to execute the startup script
CMD ["/app/startup.sh"]