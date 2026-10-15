import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
MIB = 1024 * 1024


class Settings(BaseSettings):
    """All configuration comes from environment variables (or the repo-root .env)."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",  # the .env is shared with the frontend (VITE_*)
        protected_namespaces=("settings_",),
    )

    # Server
    environment: Literal["development", "production", "test"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    backend_host: str = "127.0.0.1"
    backend_port: int = Field(8000, ge=1, le=65535)
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # Security
    rate_limit_per_minute: int = Field(30, ge=0)
    trust_proxy_headers: bool = False

    # Upload limits
    max_image_size_mb: float = Field(10, gt=0)
    max_video_size_mb: float = Field(100, gt=0)
    max_image_pixels: int = Field(40_000_000, gt=0)
    max_video_pixels: int = Field(3840 * 2160, gt=0)
    max_video_duration_sec: float = Field(60, gt=0)
    max_video_frames_analyzed: int = Field(24, ge=1, le=200)
    temp_dir: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "truthlens-uploads"
    )

    # Detector
    detector_backend: Literal["demo", "torch", "ensemble"] = "demo"
    model_path: Path | None = None
    model_meta_path: Path | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    uncertain_band_low: float = Field(0.35, ge=0, le=1)
    uncertain_band_high: float = Field(0.65, ge=0, le=1)
    request_timeout_sec: float = Field(30, gt=0)
    max_concurrent_analyses: int = Field(2, ge=1, le=16)
    result_ttl_seconds: int = Field(900, ge=1)
    result_store_max_entries: int = Field(200, ge=1)
    heatmap_enabled: bool = True
    heatmap_max_side: int = Field(256, ge=64, le=1024)

    # Video analysis
    video_timeout_sec: float = Field(120, gt=0)
    max_concurrent_video_analyses: int = Field(1, ge=1, le=8)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [o.strip().rstrip("/") for o in value.split(",") if o.strip()]
        return value

    @model_validator(mode="after")
    def check_uncertain_band(self) -> "Settings":
        if self.uncertain_band_low >= self.uncertain_band_high:
            raise ValueError("UNCERTAIN_BAND_LOW must be smaller than UNCERTAIN_BAND_HIGH")
        return self

    @property
    def max_image_bytes(self) -> int:
        return int(self.max_image_size_mb * MIB)

    @property
    def max_video_bytes(self) -> int:
        return int(self.max_video_size_mb * MIB)


@lru_cache
def get_settings() -> Settings:
    return Settings()