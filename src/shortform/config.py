"""Configuration system — loads YAML defaults + env overrides via Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


class LLMConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048
    temperature: float = 0.8


class TTSConfig(BaseModel):
    backend: str = "edge"  # edge | f5_tts
    f5_tts_cli: str = "~/.venvs/f5-tts/bin/f5-tts_infer-cli"
    voice: str = "en-US-AriaNeural"
    rate: str = "+5%"
    volume: str = "+0%"
    output_format: str = "audio-24khz-96kbitrate-mono-mp3"


class VideoConfig(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = 30
    video_bitrate: str = "8M"
    audio_bitrate: str = "192k"
    audio_sample_rate: int = 44100
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    preset: str = "medium"
    min_duration: int = 15
    max_duration: int = 60


class VisualsConfig(BaseModel):
    backend: str = "pillow"  # pillow | veo
    font_size: int = 64
    font_color: str = "#FFFFFF"
    text_margin: int = 80
    gradient_top: str = "#1a1a2e"
    gradient_bottom: str = "#16213e"
    ken_burns_zoom: float = 1.05
    crossfade_duration: float = 0.3


class MusicConfig(BaseModel):
    enabled: bool = True
    music_dir: str = "data/music"
    volume: float = 0.15  # music volume relative to narration (0.0–1.0)
    duck_volume: float = 0.08  # volume during speech
    fade_in: float = 0.5  # seconds
    fade_out: float = 1.0  # seconds


class PathsConfig(BaseModel):
    data_dir: str = "data"
    videos_dir: str = "data/videos"
    assets_dir: str = "data/assets"
    db_path: str = "data/shortform.db"

    def resolve(self, root: Path | None = None) -> dict[str, Path]:
        base = root or PROJECT_ROOT
        return {
            "data_dir": base / self.data_dir,
            "videos_dir": base / self.videos_dir,
            "assets_dir": base / self.assets_dir,
            "db_path": base / self.db_path,
        }


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    xai_api_key: str = ""
    google_gemini_api_key: str = ""
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    video: VideoConfig = VideoConfig()
    visuals: VisualsConfig = VisualsConfig()
    music: MusicConfig = MusicConfig()
    paths: PathsConfig = PathsConfig()


class StrategyConfig(BaseModel):
    name: str
    description: str = ""
    category: str = ""
    content: dict[str, Any] = {}
    prompts: dict[str, str] = {}
    topics: list[str] = []
    visuals: dict[str, Any] = {}
    tts: dict[str, Any] = {}  # backend selection + backend-specific params (ref_audio, ref_text, ...)
    music: dict[str, Any] = {}  # track, volume overrides


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_settings() -> AppSettings:
    """Load settings from default.yaml, overlaid with env vars."""
    yaml_path = CONFIG_DIR / "default.yaml"
    yaml_data: dict[str, Any] = {}
    if yaml_path.exists():
        yaml_data = load_yaml(yaml_path)
    return AppSettings(**yaml_data)


def load_strategy(name: str) -> StrategyConfig:
    """Load a named strategy from config/strategies/<name>.yaml."""
    path = CONFIG_DIR / "strategies" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Strategy not found: {path}")
    data = load_yaml(path)
    return StrategyConfig(**data)


def list_strategies() -> list[str]:
    """List available strategy names."""
    strategies_dir = CONFIG_DIR / "strategies"
    if not strategies_dir.exists():
        return []
    return [p.stem for p in strategies_dir.glob("*.yaml")]
