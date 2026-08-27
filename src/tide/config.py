"""Typed loading and validation for the TIDE YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import as_file, files
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class ModelReference:
    """A Hugging Face model pinned to an immutable revision."""

    name: str = ""
    revision: str = ""


@dataclass
class ModelsConfig:
    """Frozen model instruments used by TIDE."""

    language_model: ModelReference = field(
        default_factory=lambda: ModelReference(
            name="Qwen/Qwen2.5-1.5B",
            revision="8faed761d45a263340a0528343f099c05c9a4323",
        )
    )
    embedding_model: ModelReference = field(
        default_factory=lambda: ModelReference(
            name="BAAI/bge-base-zh-v1.5",
            revision="f03589ceff5aac7111bd60cfc7d497ca17ecac65",
        )
    )


@dataclass
class SurprisalConfig:
    """Parameters for dialogue-conditioned token surprisal."""

    max_context_tokens: int = 512


@dataclass
class ColumnConfig:
    """Input CSV column names."""

    dialogue_id: str = "dialogue_id"
    turn_id: str = "turn_id"
    turn: str = "turn"
    speaker: str = "speaker"
    text: str = "text"


@dataclass
class RuntimeConfig:
    """Local inference settings."""

    device: str = "auto"
    embedding_batch_size: int = 32
    local_files_only: bool = False
    context_template: str = "{speaker}: {text}\n"


@dataclass
class PipelineConfig:
    """Complete TIDE configuration."""

    seed: int = 42
    models: ModelsConfig = field(default_factory=ModelsConfig)
    surprisal: SurprisalConfig = field(default_factory=SurprisalConfig)
    mattr_window: int = 10
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _validate(config: PipelineConfig) -> None:
    if config.mattr_window < 1:
        raise ValueError("mattr_window must be at least 1")
    if config.surprisal.max_context_tokens < 1:
        raise ValueError("surprisal.max_context_tokens must be at least 1")
    if config.runtime.embedding_batch_size < 1:
        raise ValueError("runtime.embedding_batch_size must be at least 1")
    if config.runtime.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("runtime.device must be one of: auto, cpu, cuda, mps")
    if "{speaker}" not in config.runtime.context_template:
        raise ValueError("runtime.context_template must contain {speaker}")
    if "{text}" not in config.runtime.context_template:
        raise ValueError("runtime.context_template must contain {text}")
    for model in (config.models.language_model, config.models.embedding_model):
        if not model.name or not model.revision:
            raise ValueError("Every model must define both name and immutable revision")


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load a YAML configuration and merge it with the typed defaults."""

    schema = OmegaConf.structured(PipelineConfig)
    if path is None:
        resource = files("tide.resources").joinpath("pipeline.yaml")
        with as_file(resource) as resource_path:
            loaded = OmegaConf.load(resource_path)
    else:
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        loaded = OmegaConf.load(config_path)

    merged = OmegaConf.merge(schema, loaded)
    OmegaConf.resolve(merged)
    config = OmegaConf.to_object(merged)
    if not isinstance(config, PipelineConfig):
        raise TypeError("Configuration could not be converted to PipelineConfig")
    _validate(config)
    return config
