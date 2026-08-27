"""Local Hugging Face model backend for TIDE."""

# pyright: reportMissingImports=false

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from tide.config import PipelineConfig
from tide.metrics import aggregate_surprisal

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from torch import Tensor
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)


def _missing_models_error() -> RuntimeError:
    return RuntimeError(
        "Model dependencies are not installed. Run `pip install 'tide-dialogue[models]'` "
        "or `uv sync --extra models`."
    )


class HuggingFaceBackend:
    """Read probabilities and embeddings from frozen local model instruments."""

    def __init__(self, config: PipelineConfig, device: str | None = None) -> None:
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
        self.embedder: SentenceTransformer = SentenceTransformer(
            embedding.name,
            revision=embedding.revision,
            device=self.device,
            local_files_only=config.runtime.local_files_only,
        )
        LOGGER.info("Loading language model %s at %s", language.name, language.revision)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            language.name,
            revision=language.revision,
            local_files_only=config.runtime.local_files_only,
            use_fast=True,
        )
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.language_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            language.name,
            revision=language.revision,
            local_files_only=config.runtime.local_files_only,
            torch_dtype=dtype,
        ).to(self.device)
        self.language_model.eval()

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

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        """Return L2-normalized sentence embeddings."""

        vectors = self.embedder.encode(
            texts,
            batch_size=self.config.runtime.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=float)

    def score_surprisal(self, context: str, target: str) -> tuple[float, float]:
        """Return target-turn mean and maximum sentence surprisal in nats."""

        context_ids = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        context_ids = context_ids[-self.config.surprisal.max_context_tokens :]
        encoded_target = self.tokenizer(
            target,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        target_ids = encoded_target["input_ids"]
        offsets = encoded_target["offset_mapping"]
        if not target_ids:
            return float("nan"), float("nan")

        if not context_ids:
            seed_token = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            if seed_token is None:
                raise RuntimeError("Tokenizer has neither BOS nor EOS token for first-turn scoring")
            context_ids = [seed_token]

        input_ids: Tensor = self._torch.tensor(
            [context_ids + target_ids],
            dtype=self._torch.long,
            device=self.device,
        )
        attention_mask = self._torch.ones_like(input_ids)
        with self._torch.inference_mode():
            logits = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits[0]
        log_probabilities = self._torch.log_softmax(logits.float(), dim=-1)
        start = len(context_ids)
        surprisals = np.asarray(
            [
                -log_probabilities[start - 1 + index, token_id].item()
                for index, token_id in enumerate(target_ids)
            ],
            dtype=float,
        )
        return aggregate_surprisal(surprisals, offsets, target)
