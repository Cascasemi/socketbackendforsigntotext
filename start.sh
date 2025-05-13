#!/bin/bash
# Wait for dependencies to initialize
sleep 2

# Start the server with explicit eventlet worker
exec gunicorn --worker-class eventlet -w 1 --timeout 120 --bind 0.0.0.0:$PORT app:app