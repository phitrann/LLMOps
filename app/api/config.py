# import os

# class Config:
#     MODEL_PATH = os.getenv("MODEL_PATH", "path/to/your/model")
#     DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "/models/microsoft/Phi-3-mini-4k-instruct"
    DATABASE_URL: str = "postgresql://user:password@localhost/dbname"
    USE_ONNX: bool = True
    VERBOSE: bool = False
    TIMINGS: bool = False
    MAX_NEW_TOKENS: int = 512
    DO_SAMPLE: bool = True
    TOP_P: float = 0.9
    TEMPERATURE: float = 0.7
    MAX_LENGTH: int = 2048
    MIN_LENGTH: int = 0
    TOP_K: int = 50
    REPETITION_PENALTY: float = 1.0

    class Config:
        env_file = ".env"

