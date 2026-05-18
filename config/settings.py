from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = {"env_file": ".env"}
    app_name: str = "SpineSeg Inference API"
    weights_path: Path = BASE_DIR / "weights" / "u_net.pth"
    device: str = "cuda"
    port: int = 8080


settings = Settings()
