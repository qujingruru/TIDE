"""Dialogue-to-metrics orchestration for TIDE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tide.config import PipelineConfig, load_config
from tide.metrics import cosine_distance, lexical_entropy, mattr, segment_words

LOGGER = logging.getLogger(__name__)

METRIC_COLUMNS = [
    "lexical_entropy",
    "mattr",
    "surprisal_mean",
    "surprisal_sent_max",
    "sem_dist_partner",
    "sem_dist_self",
]


class MetricBackend(Protocol):
    """The two read-only model operations required by the deterministic pipeline."""

    def encode(self, texts: list[str]) -> NDArray[np.float64]: ...

    def score_surprisal(self, context: str, target: str) -> tuple[float, float]: ...


def _prepare_dialogues(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    columns = config.columns
    required = [columns.turn, columns.speaker, columns.text]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")
    if bool(frame[required].isna().to_numpy().any()):
        raise ValueError("Required input columns may not contain missing values")

    prepared = frame.copy()
    if columns.dialogue_id not in prepared.columns:
        prepared[columns.dialogue_id] = "dialogue_001"
    if prepared.duplicated([columns.dialogue_id, columns.turn]).any():
        raise ValueError("Turn values must be unique within each dialogue")
    return prepared.sort_values(
        [columns.dialogue_id, columns.turn],
        kind="mergesort",
    ).reset_index(drop=True)


def compute_metrics_frame(
    frame: pd.DataFrame,
    config: PipelineConfig,
    backend: MetricBackend,
) -> pd.DataFrame:
    """Compute all six TIDE metrics for every turn in a dialogue table."""

    prepared = _prepare_dialogues(frame, config)
    columns = config.columns
    rows: list[dict[str, object]] = []

    for dialogue_index, (dialogue_id, group) in enumerate(
        prepared.groupby(columns.dialogue_id, sort=False),
        start=1,
    ):
        group = group.reset_index(drop=True)
        texts = group[columns.text].astype(str).tolist()
        embeddings = backend.encode(texts)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(group):
            raise ValueError("Embedding backend returned an invalid matrix shape")

        context = ""
        previous_by_speaker: dict[str, int] = {}
        for position in range(len(group)):
            row = group.iloc[position]
            text = str(row[columns.text])
            speaker = str(row[columns.speaker])
            words = segment_words(text)
            surprisal_mean, surprisal_max = backend.score_surprisal(context, text)
            partner_positions = [
                previous_position
                for previous_speaker, previous_position in previous_by_speaker.items()
                if previous_speaker != speaker
            ]
            partner_position = max(partner_positions) if partner_positions else None
            turn_id = (
                str(row[columns.turn_id])
                if columns.turn_id in group.columns
                else f"{dialogue_id}-T{int(row[columns.turn]):03d}"
            )
            output_row: dict[str, object] = {
                "dialogue_id": dialogue_id,
                "turn_id": turn_id,
                "turn": row[columns.turn],
                "speaker": speaker,
                "n_chars": len(text),
                "lexical_entropy": lexical_entropy(words),
                "mattr": mattr(words, config.mattr_window),
                "surprisal_mean": surprisal_mean,
                "surprisal_sent_max": surprisal_max,
                "sem_dist_partner": (
                    cosine_distance(embeddings[position], embeddings[partner_position])
                    if partner_position is not None
                    else np.nan
                ),
                "sem_dist_self": (
                    cosine_distance(
                        embeddings[position],
                        embeddings[previous_by_speaker[speaker]],
                    )
                    if speaker in previous_by_speaker
                    else np.nan
                ),
            }
            if "group" in group.columns:
                output_row["group"] = row["group"]
            rows.append(output_row)
            previous_by_speaker[speaker] = position
            context += config.runtime.context_template.format(speaker=speaker, text=text)

        if dialogue_index % 10 == 0:
            LOGGER.info("Processed %d dialogues and %d turns", dialogue_index, len(rows))

    return pd.DataFrame(rows)


def compute_metrics_file(
    input_path: str | Path,
    output_path: str | Path,
    config_path: str | Path | None = None,
    backend: MetricBackend | None = None,
) -> pd.DataFrame:
    """Read a CSV, compute metrics, and write a text-free metric table."""

    input_file = Path(input_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    config = load_config(config_path)
    if backend is None:
        from tide.backends import HuggingFaceBackend

        backend = HuggingFaceBackend(config)
    frame = pd.read_csv(input_file)
    metrics = compute_metrics_frame(frame, config, backend)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_file, index=False)
    LOGGER.info("Wrote %d rows to %s", len(metrics), output_file)
    return metrics
