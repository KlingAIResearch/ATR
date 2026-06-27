from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "imgedit_pipeline.example.json"


def load_runtime_config(config_path: Optional[Union[str, os.PathLike]] = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_runtime_config(config: dict[str, Any]) -> None:
    google = config.get("google", {})
    models = config.get("models", {})

    env_map = {
        "ATR_GOOGLE_AUTH_MODE": google.get("auth_mode"),
        "GOOGLE_APPLICATION_CREDENTIALS": google.get("application_credentials"),
        "GOOGLE_CLOUD_PROJECT": google.get("project_id"),
        "GOOGLE_PROJECT_ID": google.get("project_id"),
        "GOOGLE_CLOUD_LOCATION": google.get("location"),
        "GOOGLE_LOCATION": google.get("location"),
        "GOOGLE_API_KEY": google.get("api_key"),
        "ATR_GEMINI_MODEL": models.get("gemini_model"),
        "ATR_QWEN_IMAGE_EDIT_PATH": models.get("qwen_image_edit_path"),
        "ATR_SAM3_DIR": models.get("sam3_dir"),
        "ATR_SAM3_CHECKPOINT": models.get("sam3_checkpoint"),
    }

    for key, value in env_map.items():
        if value is not None and value != "":
            os.environ[key] = str(value)


def create_genai_client():
    from google import genai

    auth_mode = os.environ.get("ATR_GOOGLE_AUTH_MODE", "vertex").lower()
    if auth_mode == "api_key":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required when ATR_GOOGLE_AUTH_MODE=api_key")
        return genai.Client(api_key=api_key)

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GOOGLE_LOCATION")
    return genai.Client(vertexai=True, project=project_id, location=location)


def get_gemini_model(default: str = "gemini-3-flash-preview") -> str:
    return os.environ.get("ATR_GEMINI_MODEL", default)


def get_qwen_image_edit_path(default: str = "./examples/models/Qwen-Image-Edit-2509") -> str:
    return os.environ.get("ATR_QWEN_IMAGE_EDIT_PATH", default)
