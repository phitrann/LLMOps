import os
import requests

# Define the directory structure
structure = {
    "app": {
        "api": ["__init__.py", "main.py", "models.py", "config.py", "utils.py"],
        "inference": ["__init__.py", "model.py", "tokenizer.py", "pipeline.py"],
        "frontend": ["__init__.py", "gradio_app.py", "components.py"],
        "db": ["__init__.py", "database.py", "models.py", "history.py"],
        "tests": ["test_api.py", "test_inference.py", "test_gradio.py", "test_db.py"],
        "main.py": None,
    },
    "scripts": {
        "download_model.py": None, 
        "start_server.sh": None, 
        "init_db.py": None,
    },
    "docker": {
        "Dockerfile": None,
        "docker-compose.yml": None,
        "k8s": ["deployment.yaml", "service.yaml"],
    },
    "docs": {},
    ".env": None,
    ".gitignore": "https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore",
    "setup.py": None,
    "requirements.txt": None,
    "README.md": None,
}

def download(path, url=None):
    response = requests.get(url)

    if response.status_code == 200:
        with open(path, 'w') as f:
            f.write(response.text)
        print(f"{os.path.basename(path)} downloaded and saved.")
    else:
        print(f"Failed to download {os.path.basename(path)}: {e}")

# Function to create directories and files
def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            if isinstance(content, list):
                os.makedirs(path, exist_ok=True)
                for file_name in content:
                    open(os.path.join(path, file_name), 'a').close()
            else:
                if content and content.startswith("http"):
                    # If content is a URL, download the content
                    download(path, content)
                else:
                    open(path, 'a').close()
    

# Generate the project structure
create_structure('.', structure)

print("Project structure generated successfully!")
