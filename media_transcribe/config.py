"""Configuration management for media-transcribe.

Config is stored as JSON at ~/.config/media-transcribe/config.json
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "media-transcribe"
CONFIG_PATH = CONFIG_DIR / "config.json"

# Whisper API hard limit is 25 MB per file; stay safely under it.
DEFAULT_CHUNK_MB = 20


@dataclass
class Config:
    openai_api_key: str = ""
    output_dir: str = str(Path.home() / "transcriptions")
    model: str = "whisper-1"
    language: str = ""  # empty = auto-detect
    chunk_mb: int = DEFAULT_CHUNK_MB
    include_timestamps: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def chunk_bytes(self) -> int:
        return self.chunk_mb * 1024 * 1024

    @property
    def output_path(self) -> Path:
        return Path(os.path.expanduser(self.output_dir))

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("extra", None)
        CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n")
        # Config contains the API key; keep it private.
        os.chmod(CONFIG_PATH, 0o600)

    @classmethod
    def load(cls) -> "Config":
        if not CONFIG_PATH.exists():
            cfg = cls()
            # Seed API key from environment if present.
            cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
            return cfg
        try:
            data = json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return cls()
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        if not cfg.openai_api_key:
            cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        return cfg
