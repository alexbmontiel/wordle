import os

from pydantic import model_validator
from pydantic_settings import BaseSettings
from pathlib import Path


class Config(BaseSettings):
    data_path: Path = Path(__file__).parent.resolve() / "data"
    words_path: Path = data_path / "five_letter_words.txt"
