# LLM Inference Project

This project uses FastAPI for the backend API, Gradio for the frontend, and Hugging Face's Transformers library for LLM inference.

## Introduction


## Project Structure
```
/LLMOps
├── /app
│   ├── /api
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI endpoints, define API routes
│   │   ├── models.py             # Pydantic models for request and response
│   │   ├── config.py             # Configuration settings for the app
│   │   └── utils.py              # Utility functions for API
│   ├── /inference
│   │   ├── __init__.py
│   │   ├── model.py              # Logic for loading and infer LLM model
│   │   └── pipeline.py           # Pipeline for processing input and output (tokenization, post-processing)
│   ├── /frontend
│   │   ├── __init__.py
│   │   └── gradio_app.py         # UI implementation using Gradio
│   ├── /db
│   │   ├── __init__.py
│   │   ├── database.py           # Database connection and session management
│   │   ├── models.py             # Pydantic models for database
│   │   ├── vectordb.py           # Vector database for storing embeddings (future work)
│   │   └── history.py            # Logic for saving and retrieving chat history
│   ├── /tests
│   │   ├── test_api.py           # Unit test for API
│   │   ├── test_inference.py     # Unit test for LLM inference
│   │   ├── test_gradio.py        # Unit test for Gradio frontend
│   │   └── test_db.py            # Unit test for database
│   └── main.py                   # Main entry point for the app
├── /scripts
│   ├── download_model.py         # Script for downloading LLM model
│   ├── start_server.sh           # Script for starting server
│   └── init_db.py                # Script for initializing database (future work)
├── /docker
│   ├── Dockerfile                # Dockerfile for building the app image
│   └── /k8s
│       ├── deployment.yaml       # Define Kubernetes Deployment
│       └── service.yaml          # Define Kubernetes Service
├── .env                          # Configuration file for environment variables
├── .gitignore                    # Files and directories to be ignored by Git
|── .dockerignore                 # Files and directories to be ignored by Docker
├── docker-compose.yml            # Docker Compose configuration
|── setup.py                      # Setup file for packaging the app
|── generate_tree.py              # Script for generating project skeleton
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

__Main components__:
1. __/app__: The main application directory, containing the core logic of the app.
    - __/api__: Holds the FastAPI endpoints, including API routes, request/response models, and utility functions.
    - __/inference__: Contains the logic for loading and inferring the LLM model, as well as the pipeline for processing input and output.
    - __/frontend__: Contains the UI implementation using Gradio.
    - __/db__: Contains the database connection and session management, Pydantic models for database, and logic for saving and retrieving chat history.
    - __/tests__: Contains unit tests for the API, LLM inference, Gradio frontend, and database.
2. __/scripts__: Scripts for downloading the LLM model, starting the server, and initializing the database (future work).
3. __/docker__: Contains the Dockerfile for building the app image and Kubernetes configuration files.

4. __.env__: Configuration file for environment variables, including the MongoDB connection string, LLM model path, and Gradio port.

5. __requirements.txt__: Python dependencies for the project.
6. __README.md__: Project documentation.

## Models
- The model used in this project is `microsoft/Phi-3-mini-4k-instruct`. Phi-3 Mini is a 3.8B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties.


## Tech Stacks
- FastAPI: A modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
- Gradio: A Python library that allows you to quickly create UIs for your machine learning models.
- Langchain: A Python library that provides components for building conversational AI systems.
- Ollama: Serving for LLM inference.
- ONNX: Open Neural Network Exchange (ONNX) is an open-source format for AI models, allowing interoperability between different frameworks.
- MongoDB: A NoSQL database that stores data in flexible, JSON-like documents.


## Requirements
### Prequisites
1. Clone the repository
```bash
git clone https://github.com/phitrann/LLMOps.git
```

2. Set up Ollama server
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

3. Pull the Phi 3 Mini model
```bash
docker exec -it ollama ollama run phi3:instruct
```

### Installation

#### Manual


1. Install dependencies
```bash
pip install -e .
pip install -r requirements.txt
```

2. Download the model (Optional)
```bash
python scripts/download_model.py
```

3. Start the MongoDB service
```
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

4. Start the server
```bash
# Start FastAPI server
uvicorn app.api.main:app --host 0.0.0.0 --port 8046 --reload

# Start Gradio app
python -m app.frontend.gradio_app # gradio app/frontend/gradio_app.py
```

#### Docker Compose (Not working yet)

```bash
docker compose up -d
```

### Test the APIs

1. Send a message to the chatbot
```
curl --no-buffer -X 'POST' 'http://localhost:8046/chat' -H 'accept: text/plain' -H 'Content-Type: application/json' -d '{"session_id": "session_1","message": "who'\''s playing in the river?"}'
```

2. Send a message to the chatbot through server sent events
```
curl --no-buffer -X 'POST' 'http://localhost:8046/chat/sse/' -H 'accept: text/event-stream' -H 'Content-Type: application/json' -d '{"session_id": "session_2", "message": "who'\''s playing in the garden?"}'
```


## Usage

- The API will be available at `http://localhost:8046`.
- The Gradio interface will be available at `http://localhost:7860`.





