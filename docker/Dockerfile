# Use an official Python runtime as a parent image
FROM python:3.9-slim

# ENV WORKSPACE_ROOT=/usr/src/app 

# RUN mkdir -p $WORKSPACE_ROOT

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends build-essential \
    python3-dev \
    curl \
    build-essential \
    && apt-get clean

WORKDIR /app

# # Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt --proxy http://192.168.100.210:3128

# CMD ["ls", "-la"]

# CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8045"]

# # Expose ports
# EXPOSE 8000


# Ensure the script is executable
RUN chmod +x ./scripts/start_server.sh

# HEALTHCHECK CMD curl --fail http://localhost:8045/health

ENTRYPOINT [ "/bin/bash", "-c" ]

# CMD ["pwd"]

# Run the start_server.sh scripts
CMD ["./scripts/start_server.sh"]
