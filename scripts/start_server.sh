#!/bin/bash


# # Read port from command line argument
# if [ $# -eq 1 ]; then
#     PORT=$1
# else
#     PORT=8045
# fi

echo "Starting server on port 8045"

docker run -d -p 27017:27017 --name mongo-llmops mongo:latest

# Start FastAPI server
uvicorn app.api.main:app --host 0.0.0.0 --port 8046 --reload

# Start Gradio app
python -m app.frontend.gradio_app # gradio app/frontend/gradio_app.py
