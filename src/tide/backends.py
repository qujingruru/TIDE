"""Local Hugging Face model backend for TIDE."""

# pyright: reportMissingImports=false

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from tide.config import PipelineConfig
from tide.metrics import aggregate_surprisal

if TYPE_CHECKING:
    from torch import Tensor

LOGGER = logging.getLogger(__name__)


def _missing_models_error() -> RuntimeError:
    return RuntimeError(
        "Model dependencies are not installed. Run `pip install 'tide-dialogue[models]'` "
        "or `uv sync --extra models`."
    )


def _configure_huggingface_offline(local_files_only: bool) -> None:
    """Make the process-level offline promise explicit before Hub imports."""

    if not local_files_only:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


@dataclass(frozen=True)
class _EmbeddingChunk:
    """One content-token chunk assigned to its source text."""

    owner: int
    token_ids: tuple[int, ...]
    weight: int


def _plan_embedding_chunks(
    tokenizer: Any,
    texts: list[str],
    *,
    max_tokens: int,
    strategy: str,
) -> list[_EmbeddingChunk]:
    """Split text into model-sized content-token chunks without decoding tokens."""

    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    content_budget = max_tokens - special_tokens
    if content_budget < 1:
        raise ValueError("embedding.max_tokens leaves no room for content tokens")
    if strategy not in {"token_weighted_chunks", "truncate"}:
        raise ValueError(f"Unknown embedding long-text strategy: {strategy}")

    chunks: list[_EmbeddingChunk] = []
    for owner, text in enumerate(texts):
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
        token_ids = [int(token_id) for token_id in encoded["input_ids"]]
        starts = [0] if strategy == "truncate" else list(range(0, len(token_ids), content_budget))
        if not starts:
            starts = [0]
        for start in starts:
            chunk_ids = tuple(token_ids[start : start + content_budget])
            chunks.append(
                _EmbeddingChunk(
                    owner=owner,
                    token_ids=chunk_ids,
                    weight=max(1, len(chunk_ids)),
                )
            )
    return chunks


def _aggregate_chunk_embeddings(
    vectors: NDArray[np.float64],
    *,
    owners: list[int],
    weights: list[int],
    text_count: int,
) -> NDArray[np.float64]:
    """Token-weight normalized chunks, average them, and normalize each text."""

    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Chunk embeddings must be a two-dimensional matrix")
    if matrix.shape[0] != len(owners) or len(owners) != len(weights):
        raise ValueError("Chunk embeddings, owners, and weights must have equal lengths")
    if text_count < 1:
        raise ValueError("text_count must be at least 1")
    if any(owner < 0 or owner >= text_count for owner in owners):
        raise ValueError("Chunk owner is outside the source-text range")
    if any(weight < 1 for weight in weights):
        raise ValueError("Chunk weights must be positive")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if bool((norms == 0).any()):
        raise ValueError("Embedding model returned a zero-vector chunk")
    normalized_chunks = matrix / norms
    aggregated = np.zeros((text_count, matrix.shape[1]), dtype=float)
    total_weights = np.zeros(text_count, dtype=float)
    for vector, owner, weight in zip(normalized_chunks, owners, weights, strict=True):
        aggregated[owner] += vector * weight
        total_weights[owner] += weight
    if bool((total_weights == 0).any()):
        raise RuntimeError("A source text did not receive an embedding chunk")
    aggregated /= total_weights[:, None]
    aggregate_norms = np.linalg.norm(aggregated, axis=1, keepdims=True)
    if bool((aggregate_norms == 0).any()):
        raise ValueError("Token-weighted chunk aggregation produced a zero vector")
    return aggregated / aggregate_norms


class HuggingFaceBackend:
    """Read probabilities and embeddings from frozen local model instruments."""

    def __init__(self, config: PipelineConfig, device: str | None = None) -> None:
        _configure_huggingface_offline(config.runtime.local_files_only)
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise _missing_models_error() from error

        self._torch = torch
        self.config = config
        self.device = self._resolve_device(device or config.runtime.device)
        self._set_seed(config.seed)

        embedding = config.models.embedding_model
        language = config.models.language_model
        LOGGER.info("Loading embedding model %s at %s", embedding.name, embedding.revision)
        # Transformers and SentenceTransformers expose version-dependent generic
        # return types. Keep the boundary dynamic and convert outputs immediately.
        self.embedder: Any = SentenceTransformer(
            embedding.name,
            revision=embedding.revision,
            device=self.device,
            local_files_only=config.runtime.local_files_only,
        )
        self.embedder.eval()
        self.embedding_tokenizer: Any = self.embedder.tokenizer
        embedding_limits = [
            limit
            for limit in [
                getattr(self.embedder, "max_seq_length", None),
                getattr(self.embedding_tokenizer, "model_max_length", None),
            ]
            if isinstance(limit, int) and 0 < limit < 1_000_000
        ]
        if embedding_limits and config.embedding.max_tokens > min(embedding_limits):
            raise ValueError(
                f"embedding.max_tokens={config.embedding.max_tokens} exceeds the embedding "
                f"model limit of {min(embedding_limits)}"
            )
        LOGGER.info("Loading language model %s at %s", language.name, language.revision)
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            language.name,
            revision=language.revision,
            local_files_only=config.runtime.local_files_only,
            use_fast=True,
        )
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        language_model: Any = AutoModelForCausalLM.from_pretrained(
            language.name,
            revision=language.revision,
            local_files_only=config.runtime.local_files_only,
            dtype=dtype,
        )
        self.language_model: Any = language_model.to(self.device)
        self.language_model.eval()
        self.max_sequence_tokens = self._infer_max_sequence_tokens()

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            if requested == "cuda" and not self._torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            if requested == "mps" and not self._torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is not available")
            return requested
        if self._torch.cuda.is_available():
            return "cuda"
        if self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)

    def _infer_max_sequence_tokens(self) -> int | None:
        """Return the smallest credible tokenizer/model position limit."""

        candidates: list[int] = []
        model_config = getattr(self.language_model, "config", None)
        for value in [
            getattr(model_config, "max_position_embeddings", None),
            getattr(self.tokenizer, "model_max_length", None),
        ]:
            if isinstance(value, int) and 0 < value < 1_000_000:
                candidates.append(value)
        return min(candidates) if candidates else None

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        """Return full-turn embeddings with explicit long-text handling."""

        chunks = _plan_embedding_chunks(
            self.embedding_tokenizer,
            texts,
            max_tokens=self.config.embedding.max_tokens,
            strategy=self.config.embedding.long_text_strategy,
        )
        chunk_vectors: list[NDArray[np.float64]] = []
        batch_size = self.config.runtime.embedding_batch_size
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            feature_rows: list[dict[str, list[int]]] = []
            for chunk in batch:
                content_ids = list(chunk.token_ids)
                cls_token_id = getattr(self.embedding_tokenizer, "cls_token_id", None)
                sep_token_id = getattr(self.embedding_tokenizer, "sep_token_id", None)
                if cls_token_id is None or sep_token_id is None:
                    raise RuntimeError(
                        "The pinned embedding tokenizer must define CLS and SEP token IDs"
                    )
                input_ids = [int(cls_token_id), *content_ids, int(sep_token_id)]
                if len(input_ids) > self.config.embedding.max_tokens:
                    raise RuntimeError("An embedding chunk exceeds embedding.max_tokens")
                feature_rows.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                    }
                )
            features = self.embedding_tokenizer.pad(
                feature_rows,
                padding=True,
                return_tensors="pt",
            )
            features = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in features.items()
            }
            with self._torch.inference_mode():
                output = self.embedder.forward(features)
                embeddings = self._torch.nn.functional.normalize(
                    output["sentence_embedding"].float(),
                    p=2,
                    dim=1,
                )
            chunk_vectors.append(np.asarray(embeddings.detach().cpu().numpy(), dtype=float))
        matrix = np.vstack(chunk_vectors)
        return _aggregate_chunk_embeddings(
            matrix,
            owners=[chunk.owner for chunk in chunks],
            weights=[chunk.weight for chunk in chunks],
            text_count=len(texts),
        )

    def _prepare_surprisal_request(
        self,
        context: str,
        target: str,
    ) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        boundary = len(context)
        encoded = self.tokenizer(
            context + target,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = list(encoded["input_ids"])
        combined_offsets = list(encoded["offset_mapping"])
        if len(token_ids) != len(combined_offsets):
            raise RuntimeError("Tokenizer returned misaligned token IDs and offsets")
        crossing_tokens = [
            (index, start, end)
            for index, (start, end) in enumerate(combined_offsets)
            if start < boundary < end
        ]
        if crossing_tokens:
            raise ValueError(
                "A tokenizer token crosses the context-target boundary. "
                "Use a context/target separator that the pinned tokenizer does not merge."
            )

        target_start = next(
            (index for index, (_start, end) in enumerate(combined_offsets) if end > boundary),
            len(token_ids),
        )
        context_ids = token_ids[:target_start]
        target_ids = token_ids[target_start:]
        offsets = [
            (max(0, start - boundary), max(0, end - boundary))
            for start, end in combined_offsets[target_start:]
        ]
        if target and not target_ids:
            raise RuntimeError("Tokenizer did not return target tokens")
        context_ids = context_ids[-self.config.surprisal.max_context_tokens :]

        if not context_ids:
            seed_token = self.tokenizer.bos_token_id
            if seed_token is None:
                seed_token = self.tokenizer.eos_token_id
            if seed_token is None:
                raise RuntimeError("Tokenizer has neither BOS nor EOS token for first-turn scoring")
            context_ids = [seed_token]
        sequence_length = len(context_ids) + len(target_ids)
        if self.max_sequence_tokens is not None and sequence_length > self.max_sequence_tokens:
            raise ValueError(
                f"The context-target sequence contains {sequence_length} tokens after "
                f"context truncation, exceeding the model position limit of "
                f"{self.max_sequence_tokens}. TIDE does not truncate target text."
            )
        return context_ids, target_ids, offsets

    def score_surprisal(self, context: str, target: str) -> tuple[float, float]:
        """Return target-turn mean and maximum sentence surprisal in nats."""

        return self.score_surprisal_batch([(context, target)])[0]

    def score_surprisal_batch(
        self,
        requests: list[tuple[str, str]],
    ) -> list[tuple[float, float]]:
        """Score context-target pairs in deterministic small padded batches."""

        prepared = [
            (*self._prepare_surprisal_request(context, target), target)
            for context, target in requests
        ]
        token_budget = self.config.runtime.language_token_budget
        oversized_requests = [
            len(context_ids) + len(target_ids)
            for context_ids, target_ids, _offsets, _target in prepared
            if len(context_ids) + len(target_ids) > token_budget
        ]
        if oversized_requests:
            raise ValueError(
                f"A context-target request contains {max(oversized_requests)} tokens, "
                f"exceeding the configured language token budget of {token_budget}. "
                "Increase runtime.language_token_budget or shorten the target text."
            )
        indexed = sorted(
            enumerate(prepared),
            key=lambda item: len(item[1][0]) + len(item[1][1]),
        )
        batches: list[list[tuple[int, Any]]] = []
        batch: list[tuple[int, Any]] = []
        maximum_batch_size = self.config.runtime.language_batch_size
        for item in indexed:
            candidate = [*batch, item]
            candidate_maximum = max(
                len(prepared_item[0]) + len(prepared_item[1])
                for _original_index, prepared_item in candidate
            )
            if batch and (
                len(candidate) > maximum_batch_size
                or candidate_maximum * len(candidate) > token_budget
            ):
                batches.append(batch)
                batch = [item]
            else:
                batch = candidate
        if batch:
            batches.append(batch)

        results: list[tuple[float, float] | None] = [None] * len(prepared)
        for indexed_batch in batches:
            original_indices = [original_index for original_index, _item in indexed_batch]
            prepared_batch = [item for _original_index, item in indexed_batch]
            maximum_length = max(
                len(context_ids) + len(target_ids)
                for context_ids, target_ids, _, _ in prepared_batch
            )
            pad_token = self.tokenizer.pad_token_id
            if pad_token is None:
                pad_token = self.tokenizer.eos_token_id
            if pad_token is None:
                pad_token = 0
            input_ids: Tensor = self._torch.full(
                (len(prepared_batch), maximum_length),
                pad_token,
                dtype=self._torch.long,
                device=self.device,
            )
            attention_mask: Tensor = self._torch.zeros_like(input_ids)
            for row_index, (context_ids, target_ids, _offsets, _target) in enumerate(
                prepared_batch
            ):
                sequence = context_ids + target_ids
                if sequence:
                    input_ids[row_index, : len(sequence)] = self._torch.tensor(
                        sequence,
                        dtype=self._torch.long,
                        device=self.device,
                    )
                    attention_mask[row_index, : len(sequence)] = 1

            base_model = getattr(self.language_model, "model", None)
            output_embeddings = getattr(
                self.language_model,
                "get_output_embeddings",
                lambda: None,
            )()
            if callable(base_model) and callable(output_embeddings):
                with self._torch.inference_mode():
                    base_output: Any = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden_states: Any = base_output.last_hidden_state
                for row_index, (context_ids, target_ids, offsets, target) in enumerate(
                    prepared_batch
                ):
                    original_index = original_indices[row_index]
                    if not target_ids:
                        results[original_index] = (float("nan"), float("nan"))
                        continue
                    start = len(context_ids)
                    target_hidden = hidden_states[
                        row_index,
                        start - 1 : start - 1 + len(target_ids),
                    ]
                    target_tensor = self._torch.tensor(
                        target_ids,
                        dtype=self._torch.long,
                        device=self.device,
                    )
                    surprisal_chunks: list[NDArray[np.float64]] = []
                    logit_chunk_size = self.config.runtime.language_logit_chunk_size
                    for chunk_start in range(0, len(target_ids), logit_chunk_size):
                        chunk_end = min(chunk_start + logit_chunk_size, len(target_ids))
                        with self._torch.inference_mode():
                            chunk_logits: Any = output_embeddings(
                                target_hidden[chunk_start:chunk_end]
                            )
                            chunk_log_probabilities = self._torch.log_softmax(
                                chunk_logits.float(),
                                dim=-1,
                            )
                            chunk_indices = self._torch.arange(
                                chunk_end - chunk_start,
                                device=self.device,
                            )
                            chunk_targets = target_tensor[chunk_start:chunk_end]
                            chunk_surprisals = -chunk_log_probabilities[
                                chunk_indices,
                                chunk_targets,
                            ]
                        surprisal_chunks.append(
                            np.asarray(
                                chunk_surprisals.detach().cpu().numpy(),
                                dtype=float,
                            )
                        )
                    surprisals = np.concatenate(surprisal_chunks)
                    results[original_index] = aggregate_surprisal(surprisals, offsets, target)
                del hidden_states
            else:
                # Compatibility path for simple custom/fake causal language models.
                with self._torch.inference_mode():
                    logits = self.language_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits
                for row_index, (context_ids, target_ids, offsets, target) in enumerate(
                    prepared_batch
                ):
                    original_index = original_indices[row_index]
                    if not target_ids:
                        results[original_index] = (float("nan"), float("nan"))
                        continue
                    start = len(context_ids)
                    target_logits = logits[
                        row_index,
                        start - 1 : start - 1 + len(target_ids),
                    ]
                    log_probabilities = self._torch.log_softmax(target_logits.float(), dim=-1)
                    token_indices = self._torch.arange(len(target_ids), device=self.device)
                    target_tensor = self._torch.tensor(
                        target_ids,
                        dtype=self._torch.long,
                        device=self.device,
                    )
                    surprisals = (
                        -log_probabilities[token_indices, target_tensor]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(float)
                    )
                    results[original_index] = aggregate_surprisal(surprisals, offsets, target)
                del logits
            if self.device == "mps":
                self._torch.mps.empty_cache()
            elif self.device == "cuda":
                self._torch.cuda.empty_cache()
        if any(result is None for result in results):
            raise RuntimeError("A batched surprisal request did not produce a result")
        return [result for result in results if result is not None]
