import os
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: str = Field(
        default="./models/microsoft/Phi-3-mini-4k-instruct",
        description="The path to the model to use",
    )
    BACKEND_TYPE: str = Field(
        default="transformers",
        description="The backend type to use, options: llama.cpp, Ollama, transformers, onnx",
    )
    DATABASE_URL: str = "postgresql://user:password@localhost/dbname"
    HOST: str = Field(default="localhost", description="API address")
    PORT: int = Field(default=8046, description="API port")
    CHAT_HISTORY_LIMIT: int = Field(default=10, description="Number of recent messages to include in chat history")
    
    VERBOSE: bool = False
    TIMINGS: bool = False
    MAX_TOKENS: int = 2048
    MAX_NEW_TOKENS: int = 1024
    DO_SAMPLE: bool = True
    TOP_P: float = 0.9
    TEMPERATURE: float = 0.7
    MAX_LENGTH: int = 2048
    MIN_LENGTH: int = 0
    TOP_K: int = 50
    REPETITION_PENALTY: float = 1.0
    
    LOAD_IN_8BITS: bool = False

    class Config:
        env_file = ".env"

