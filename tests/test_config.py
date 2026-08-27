from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest

from tide.config import load_config


def test_default_config_pins_models_and_seed() -> None:
    config = load_config()
    assert config.seed == 42
    assert config.models.language_model.name == "Qwen/Qwen2.5-1.5B"
    assert len(config.models.language_model.revision) == 40
    assert config.models.embedding_model.name == "BAAI/bge-base-zh-v1.5"
    assert len(config.models.embedding_model.revision) == 40


def test_repository_config_matches_packaged_default() -> None:
    repository_config = Path(__file__).parents[1] / "config" / "pipeline.yaml"
    packaged_config = files("tide.resources").joinpath("pipeline.yaml")
    assert repository_config.read_text(encoding="utf-8") == packaged_config.read_text(
        encoding="utf-8"
    )


def test_invalid_config_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("mattr_window: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mattr_window"):
        load_config(invalid)


def test_missing_config_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(tmp_path / "missing.yaml")
