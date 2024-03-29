#!/bin/bash

# Start the Prefect server in the background
prefect server start &

# Start the Prefect worker with a managed pool named 'my-managed-pool' in the background
prefect worker start --pool my-managed-pool &

# Run the Python script create_deployment.py
python prefect-dags/create_deployment.py
