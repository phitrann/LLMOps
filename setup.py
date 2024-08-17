from setuptools import setup, find_packages

setup(
    name="llmops",
    version="0.1.0",
    author="Phi Tran",
    author_email="hoanganh6758@gmail.com",
    description="A project for LLM inference using FastAPI, Gradio, and Transformers.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/phitrann/LLMOps",  # Update with your repository URL
    license='MIT',
    packages=find_packages(include=["app", "app.*"]),
    install_requires=[''],
    entry_points={
        "console_scripts": [
            "start-server=scripts.start_server:main",
            "download-model=scripts.download_model:main",
            "init-db=scripts.init_db:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
