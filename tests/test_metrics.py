from __future__ import annotations

import math

import numpy as np
import pytest

from tide.metrics import (
    aggregate_surprisal,
    cosine_distance,
    count_non_whitespace_characters,
    lexical_entropy,
    mattr,
    segment_words,
    sentence_spans,
)


def test_sentence_spans_support_chinese_and_english_punctuation() -> None:
    text = "第一句。第二句！Third sentence?"
    assert [text[start:end] for start, end in sentence_spans(text)] == [
        "第一句。",
        "第二句！",
        "Third sentence?",
    ]


def test_lexical_entropy_has_known_answer() -> None:
    assert lexical_entropy(["甲", "甲", "乙", "乙"]) == pytest.approx(math.log(2))
    assert lexical_entropy([]) == 0.0
    assert lexical_entropy(["甲"]) == 0.0


def test_segment_words_normalizes_case_width_and_discards_symbols() -> None:
    assert segment_words("ＡＩ AI ai，？！ １２3\n观点") == [  # noqa: RUF001
        "ai",
        "ai",
        "ai",
        "123",
        "观点",
    ]


def test_character_count_ignores_formatting_whitespace_after_nfkc() -> None:
    assert count_non_whitespace_characters("Ａ I\n观点\t！") == 5  # noqa: RUF001


def test_mattr_has_known_sliding_window_answer() -> None:
    words = ["a", "a", "b", "c"]
    assert mattr(words, window=3) == pytest.approx(((2 / 3) + 1.0) / 2)
    assert mattr(["a", "b"], window=10) == 1.0
    assert np.isnan(mattr([], window=10))
    with pytest.raises(ValueError, match="at least 1"):
        mattr(words, window=0)


def test_cosine_distance_has_known_geometry() -> None:
    assert cosine_distance([1, 0], [1, 0]) == pytest.approx(0.0)
    assert cosine_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    assert cosine_distance([1, 0], [-1, 0]) == pytest.approx(2.0)
    assert np.isnan(cosine_distance([0, 0], [1, 0]))
    with pytest.raises(ValueError, match="same shape"):
        cosine_distance([1, 0], [1, 0, 0])


def test_surprisal_aggregation_has_known_sentence_maximum() -> None:
    text = "甲乙。丙丁！"
    values = [1.0, 1.0, 3.0, 3.0]
    offsets = [(0, 1), (1, 2), (3, 4), (4, 5)]
    turn_mean, sentence_max = aggregate_surprisal(values, offsets, text)
    assert turn_mean == pytest.approx(2.0)
    assert sentence_max == pytest.approx(3.0)


def test_surprisal_aggregation_validates_inputs() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        aggregate_surprisal([[1.0]], [(0, 1)], "甲")
    with pytest.raises(ValueError, match="equal lengths"):
        aggregate_surprisal([1.0], [], "甲")
    assert all(np.isnan(value) for value in aggregate_surprisal([], [], ""))
