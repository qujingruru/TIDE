"""Dialogue-to-metrics orchestration for TIDE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tide.config import PipelineConfig, load_config
from tide.metrics import (
    cosine_distance,
    count_non_whitespace_characters,
    lexical_entropy,
    mattr,
    segment_words,
)

LOGGER = logging.getLogger(__name__)

METRIC_COLUMNS = [
    "lexical_entropy",
    "mattr",
    "surprisal_mean",
    "surprisal_sent_max",
    "sem_dist_partner",
    "sem_dist_self",
    "self_novelty",
    "delta_surprisal",
    "peak_break",
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
    if bool(frame[columns.text].astype(str).str.strip().eq("").to_numpy().any()):
        raise ValueError("Turn text may not be empty or whitespace only")

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
    """Compute all nine TIDE readouts for every turn in a dialogue table."""

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

        contexts: list[str] = []
        context_speaker_labels: dict[str, str] = {}
        context = ""
        for position in range(len(group)):
            row = group.iloc[position]
            speaker = str(row[columns.speaker])
            if config.runtime.normalize_speakers:
                speaker_label = context_speaker_labels.setdefault(
                    speaker,
                    f"S{len(context_speaker_labels) + 1:02d}",
                )
            else:
                speaker_label = speaker
            contexts.append(
                context + config.runtime.target_prefix_template.format(speaker=speaker_label)
            )
            context += config.runtime.context_template.format(
                speaker=speaker_label,
                text=str(row[columns.text]),
            )
        batch_scorer = getattr(backend, "score_surprisal_batch", None)
        if callable(batch_scorer):
            surprisal_scores = cast(
                list[tuple[float, float]],
                batch_scorer(list(zip(contexts, texts, strict=True))),
            )
        else:
            surprisal_scores = [
                backend.score_surprisal(turn_context, text)
                for turn_context, text in zip(contexts, texts, strict=True)
            ]
        if len(surprisal_scores) != len(group):
            raise ValueError("Surprisal backend returned the wrong number of scores")

        previous_by_speaker: dict[str, int] = {}
        positions_by_speaker: dict[str, list[int]] = {}
        surprisal_by_speaker: dict[str, list[float]] = {}
        for position in range(len(group)):
            row = group.iloc[position]
            text = str(row[columns.text])
            speaker = str(row[columns.speaker])
            words = segment_words(text)
            surprisal_mean, surprisal_max = surprisal_scores[position]
            partner_positions = [
                previous_position
                for previous_speaker, previous_position in previous_by_speaker.items()
                if previous_speaker != speaker
            ]
            partner_position = max(partner_positions) if partner_positions else None
            own_positions = positions_by_speaker.get(speaker, [])
            own_surprisals = surprisal_by_speaker.get(speaker, [])
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
                "n_chars": count_non_whitespace_characters(text),
                "n_words": len(words),
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
                "self_novelty": (
                    min(
                        cosine_distance(embeddings[position], embeddings[prior_position])
                        for prior_position in own_positions
                    )
                    if own_positions
                    else np.nan
                ),
                "delta_surprisal": (
                    surprisal_mean - own_surprisals[-1] if own_surprisals else np.nan
                ),
                "peak_break": (surprisal_mean - max(own_surprisals) if own_surprisals else np.nan),
            }
            if "group" in group.columns:
                output_row["group"] = row["group"]
            rows.append(output_row)
            previous_by_speaker[speaker] = position
            positions_by_speaker.setdefault(speaker, []).append(position)
            surprisal_by_speaker.setdefault(speaker, []).append(surprisal_mean)

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
