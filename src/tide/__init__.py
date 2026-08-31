"""TIDE: Turn-level Information-theoretic Dialogue Evaluation."""

from tide.config import PipelineConfig, load_config
from tide.metrics import (
    aggregate_surprisal,
    cosine_distance,
    count_non_whitespace_characters,
    lexical_entropy,
    mattr,
    sentence_spans,
)
from tide.pipeline import compute_metrics_file, compute_metrics_frame

__all__ = [
    "PipelineConfig",
    "aggregate_surprisal",
    "compute_metrics_file",
    "compute_metrics_frame",
    "cosine_distance",
    "count_non_whitespace_characters",
    "lexical_entropy",
    "load_config",
    "mattr",
    "sentence_spans",
]

__version__ = "0.1.0"
