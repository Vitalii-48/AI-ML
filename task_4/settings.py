from pydantic_settings import BaseSettings, SettingsConfigDict
from constants import SCRIPT_DIR

class Settings(BaseSettings):
    groq_api_key: str

    model_config = SettingsConfigDict(
        env_file=SCRIPT_DIR.parent / ".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
