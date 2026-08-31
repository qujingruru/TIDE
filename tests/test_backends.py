from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from tide.backends import (
    HuggingFaceBackend,
    _aggregate_chunk_embeddings,
    _configure_huggingface_offline,
    _plan_embedding_chunks,
)
from tide.config import load_config


class BoundaryTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, Any]:
        encodings = {
            "S01:\n我。": {
                "input_ids": [10, 20, 21],
                "offset_mapping": [(0, 5), (5, 6), (6, 7)],
            },
            "A\nB\n目标": {
                "input_ids": [11, 12, 13, 30, 31],
                "offset_mapping": [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6)],
            },
            "abcdef": {
                "input_ids": [99],
                "offset_mapping": [(0, 6)],
            },
            "A目标": {
                "input_ids": [11, 30, 31],
                "offset_mapping": [(0, 1), (1, 2), (2, 3)],
            },
            "": {"input_ids": [], "offset_mapping": []},
        }
        return encodings[text]


class ChunkTokenizer:
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert pair is False
        return 2

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, Any]:
        return {"input_ids": list(range(1, len(text) + 1))}


def _backend(
    max_context_tokens: int = 512,
    max_sequence_tokens: int = 32_768,
) -> HuggingFaceBackend:
    backend = object.__new__(HuggingFaceBackend)
    backend.tokenizer = BoundaryTokenizer()
    backend.config = load_config()
    backend.config.surprisal.max_context_tokens = max_context_tokens
    backend.max_sequence_tokens = max_sequence_tokens
    return backend


def test_prepare_surprisal_uses_canonical_joint_tokenization() -> None:
    context_ids, target_ids, offsets = _backend()._prepare_surprisal_request(
        "S01:\n",
        "我。",
    )
    assert context_ids == [10]
    assert target_ids == [20, 21]
    assert offsets == [(0, 1), (1, 2)]


def test_prepare_surprisal_truncates_only_context_tokens() -> None:
    context_ids, target_ids, offsets = _backend(max_context_tokens=2)._prepare_surprisal_request(
        "A\nB\n",
        "目标",
    )
    assert context_ids == [12, 13]
    assert target_ids == [30, 31]
    assert offsets == [(0, 1), (1, 2)]


def test_prepare_surprisal_seeds_an_empty_sequence() -> None:
    context_ids, target_ids, offsets = _backend()._prepare_surprisal_request("", "")
    assert context_ids == [1]
    assert target_ids == []
    assert offsets == []


def test_prepare_surprisal_rejects_token_crossing_context_target_boundary() -> None:
    with pytest.raises(ValueError, match="crosses the context-target boundary"):
        _backend()._prepare_surprisal_request("abc", "def")


def test_prepare_surprisal_rejects_sequence_beyond_model_limit() -> None:
    with pytest.raises(ValueError, match="model position limit"):
        _backend(max_sequence_tokens=2)._prepare_surprisal_request("A", "目标")


def test_embedding_chunk_plan_uses_full_content_without_overlap() -> None:
    chunks = _plan_embedding_chunks(
        ChunkTokenizer(),
        ["abcdefghij", "xy"],
        max_tokens=6,
        strategy="token_weighted_chunks",
    )
    assert [(chunk.owner, list(chunk.token_ids), chunk.weight) for chunk in chunks] == [
        (0, [1, 2, 3, 4], 4),
        (0, [5, 6, 7, 8], 4),
        (0, [9, 10], 2),
        (1, [1, 2], 2),
    ]


def test_embedding_chunk_plan_can_expose_truncation_as_a_specification() -> None:
    chunks = _plan_embedding_chunks(
        ChunkTokenizer(),
        ["abcdefghij"],
        max_tokens=6,
        strategy="truncate",
    )
    assert [(list(chunk.token_ids), chunk.weight) for chunk in chunks] == [([1, 2, 3, 4], 4)]


def test_embedding_chunk_aggregation_is_token_weighted_and_normalized() -> None:
    aggregated = _aggregate_chunk_embeddings(
        np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
        owners=[0, 0, 1],
        weights=[3, 1, 2],
        text_count=2,
    )
    assert aggregated[0] == pytest.approx([3 / np.sqrt(10), 1 / np.sqrt(10)])
    assert aggregated[1] == pytest.approx([0.0, 1.0])


def test_offline_configuration_sets_huggingface_process_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"]:
        monkeypatch.delenv(variable, raising=False)
    _configure_huggingface_offline(True)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


class BatchTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, Any]:
        encodings = {
            "c1t1x": {
                "input_ids": [4, 5, 6],
                "offset_mapping": [(0, 2), (2, 4), (4, 5)],
            },
            "ccccu": {
                "input_ids": [7, 8, 9],
                "offset_mapping": [(0, 2), (2, 4), (4, 5)],
            },
        }
        return encodings[text]


class NextTokenModel:
    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        del attention_mask
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros((batch_size, sequence_length, 16), dtype=torch.float32)
        for row in range(batch_size):
            for position in range(sequence_length - 1):
                logits[row, position, input_ids[row, position + 1]] = 8.0
        return SimpleNamespace(logits=logits)


class HiddenNextTokenBase:
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> Any:
        del attention_mask
        assert use_cache is False
        assert return_dict is True
        batch_size, sequence_length = input_ids.shape
        hidden = torch.zeros((batch_size, sequence_length, 16), dtype=torch.float32)
        for row in range(batch_size):
            for position in range(sequence_length - 1):
                hidden[row, position, input_ids[row, position + 1]] = 1.0
        return SimpleNamespace(last_hidden_state=hidden)


class HiddenNextTokenModel:
    def __init__(self) -> None:
        self.model = HiddenNextTokenBase()
        self.output = torch.nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            self.output.weight.copy_(8.0 * torch.eye(16))

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.output


def _batch_backend(token_budget: int = 32) -> HuggingFaceBackend:
    backend = object.__new__(HuggingFaceBackend)
    backend._torch = cast(Any, torch)
    backend.device = "cpu"
    backend.tokenizer = BatchTokenizer()
    backend.language_model = NextTokenModel()
    backend.config = load_config()
    backend.config.runtime.language_batch_size = 2
    backend.config.runtime.language_token_budget = token_budget
    backend.max_sequence_tokens = 32
    return backend


def _hidden_batch_backend(token_budget: int = 32) -> HuggingFaceBackend:
    backend = _batch_backend(token_budget)
    backend.language_model = HiddenNextTokenModel()
    backend.config.runtime.language_logit_chunk_size = 1
    return backend


def test_surprisal_batch_matches_scalar_across_padding_lengths() -> None:
    backend = _batch_backend()
    requests = [("c1", "t1x"), ("cccc", "u")]
    batched = backend.score_surprisal_batch(requests)
    scalar = [backend.score_surprisal(context, target) for context, target in requests]
    assert batched == pytest.approx(scalar)
    assert batched[0][0] < 0.01
    assert batched[0][1] < 0.01


def test_hidden_state_logit_chunks_match_full_logit_compatibility_path() -> None:
    requests = [("c1", "t1x"), ("cccc", "u")]
    full_logits = _batch_backend().score_surprisal_batch(requests)
    chunked_logits = _hidden_batch_backend().score_surprisal_batch(requests)
    assert chunked_logits == pytest.approx(full_logits)


def test_surprisal_batch_rejects_single_request_over_token_budget() -> None:
    with pytest.raises(ValueError, match="language token budget"):
        _batch_backend(token_budget=2).score_surprisal_batch([("c1", "t1x")])
