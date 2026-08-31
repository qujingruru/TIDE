"""Deterministic metric functions used by the TIDE pipeline."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

import jieba
import numpy as np
from numpy.typing import ArrayLike, NDArray

SENTENCE_PATTERN = re.compile(r"[^。！？；!?.;]+[。！？；!?.;]*")
WHITESPACE_PATTERN = re.compile(r"\s+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return non-empty Chinese or English sentence spans as character offsets."""

    return [
        (match.start(), match.end())
        for match in SENTENCE_PATTERN.finditer(text)
        if match.group().strip()
    ]


def segment_words(text: str) -> list[str]:
    """Return normalized lexical tokens while excluding punctuation and symbols."""

    normalized = WHITESPACE_PATTERN.sub(" ", unicodedata.normalize("NFKC", text)).strip()
    tokens: list[str] = []
    for raw_token in jieba.lcut(normalized):
        token = raw_token.strip().casefold()
        if token and any(unicodedata.category(character)[0] in {"L", "N"} for character in token):
            tokens.append(token)
    return tokens


def count_non_whitespace_characters(text: str) -> int:
    """Count NFKC-normalized characters other than formatting whitespace."""

    normalized = unicodedata.normalize("NFKC", text)
    return sum(not character.isspace() for character in normalized)


def lexical_entropy(words: Sequence[str]) -> float:
    """Compute Shannon entropy of a token-frequency distribution in nats."""

    if len(words) < 2:
        return 0.0
    _, counts = np.unique(np.asarray(words, dtype=object), return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def mattr(words: Sequence[str], window: int = 10) -> float:
    """Compute the moving-average type-token ratio."""

    if window < 1:
        raise ValueError("window must be at least 1")
    token_count = len(words)
    if token_count == 0:
        return float("nan")
    if token_count <= window:
        return len(set(words)) / token_count
    scores = [
        len(set(words[start : start + window])) / window
        for start in range(token_count - window + 1)
    ]
    return float(np.mean(scores))


def cosine_distance(left: ArrayLike, right: ArrayLike) -> float:
    """Return one minus cosine similarity, or NaN for a zero vector."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError("Vectors must have the same shape")
    denominator = np.linalg.norm(left_array) * np.linalg.norm(right_array)
    if denominator == 0:
        return float("nan")
    similarity = float(np.dot(left_array, right_array) / denominator)
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def aggregate_surprisal(
    token_surprisals: ArrayLike,
    offsets: Sequence[tuple[int, int]],
    text: str,
) -> tuple[float, float]:
    """Aggregate token surprisals into turn mean and maximum sentence mean."""

    values: NDArray[np.float64] = np.asarray(token_surprisals, dtype=float)
    if values.ndim != 1:
        raise ValueError("token_surprisals must be one-dimensional")
    if len(values) != len(offsets):
        raise ValueError("token_surprisals and offsets must have equal lengths")
    if len(values) == 0:
        return float("nan"), float("nan")

    turn_mean = float(values.mean())
    spans = sentence_spans(text)
    if not spans:
        return turn_mean, turn_mean

    sentence_values: list[list[float]] = [[] for _ in spans]
    for value, (start, _end) in zip(values, offsets, strict=True):
        for sentence_index, (sentence_start, sentence_end) in enumerate(spans):
            if sentence_start <= start < sentence_end:
                sentence_values[sentence_index].append(float(value))
                break

    sentence_means = [float(np.mean(group)) for group in sentence_values if group]
    if not sentence_means:
        return turn_mean, turn_mean
    return turn_mean, max(sentence_means)
