# TIDE

**TIDE (Turn-level Information-theoretic Dialogue Evaluation)** is an open, local pipeline for measuring turn-level properties of dialogue. It uses language models only as read-out instruments: Qwen2.5-1.5B supplies next-token probabilities and BGE-base-zh-v1.5 supplies sentence embeddings. All metric calculation and statistical analysis are deterministic code. TIDE does not call a commercial or generative judging API.

TIDE accompanies the manuscript *What Can Information-Theoretic Metrics Tell Us about Student Thinking in Dialogue? A Design Framework, Emerging Boundary Evidence, and an Open Pipeline*.

> [!IMPORTANT]
> **Paper reproduction:** use the immutable
> [`v0.1.0` release](https://github.com/qujingruru/TIDE/releases/tag/v0.1.0)
> at commit `a7e65f020b6115f71822bdc25dbff27ad3a8726f`. The default branch may
> continue to evolve as TIDE is extended; manuscript results should be reproduced from the
> tagged release.

## What is included

- A pip-installable Python package and `tide` command-line interface.
- Six principal turn-level metrics: lexical entropy, MATTR, mean surprisal, maximum sentence-level surprisal, semantic distance to the preceding partner turn, and semantic distance to the speaker's previous turn.
- Three explicitly derived, speaker-referenced readouts: distance to the most similar prior self turn, change from the preceding self turn's surprisal, and change from the speaker's prior surprisal peak.
- Immutable Hugging Face model revisions and all parameters in one YAML configuration.
- A fully synthetic Chinese debate dataset for end-to-end use.
- Unit tests with constructed inputs of known answers.
- The turn-level creative- and critical-thinking rubrics used in the study.
- A de-identified numerical table with metrics, adjudicated ratings, and paired
  rater values for reliability analysis, but no dialogue text.
- Online Table S1 and a machine-readable CSV reporting dialogue-bootstrap
  human-rating reliability estimates.
- Reproduction code for the complete validation report and both data-driven paper figures.
- An additional partial-correlation heatmap for inspecting construct patterns.

## Installation

TIDE requires Python 3.10 or later. [`uv`](https://docs.astral.sh/uv/) is recommended.

```bash
git clone https://github.com/qujingruru/TIDE.git
cd TIDE
uv sync --extra models --extra analysis
```

An ordinary pip installation from GitHub is also supported:

```bash
python -m pip install "tide-dialogue[models,analysis] @ git+https://github.com/qujingruru/TIDE.git"
```

The first metric run downloads the two pinned model snapshots from Hugging Face. Inference then runs locally on CUDA, Apple Silicon MPS, or CPU. To require an existing local Hugging Face cache and prohibit downloads, set `runtime.local_files_only: true` in `config/pipeline.yaml`.

## Model instruments

| Role | Hugging Face model | Pinned revision |
|---|---|---|
| Next-token probabilities | `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` |
| Chinese sentence embeddings | `BAAI/bge-base-zh-v1.5` | `f03589ceff5aac7111bd60cfc7d497ca17ecac65` |

The language model is not prompted to rate, judge, or interpret a turn. TIDE reads probabilities and embeddings only.

## Quick start

The minimal input columns are `turn`, `speaker`, and `text`. `dialogue_id` and `turn_id` are optional; see `data/README.md` for the full schema.

```bash
uv run --extra models tide compute data/demo/chinese_debates.csv \
  --config config/pipeline.yaml \
  --output outputs/demo_metrics.csv
```

The output deliberately omits dialogue text. Each row contains identifiers,
character and segmented-word length, the six principal metrics, and the three
speaker-referenced readouts. TIDE reports statistical properties of
language; it does not output creativity or critical-thinking scores.

```text
dialogue_id, turn_id, turn, speaker, n_chars, n_words,
lexical_entropy, mattr, surprisal_mean, surprisal_sent_max,
sem_dist_partner, sem_dist_self, self_novelty, delta_surprisal, peak_break
```

The first semantic distance in a dialogue and the first same-speaker distance are undefined and are written as missing values.

## Configuration

All model identifiers, immutable revisions, the random seed, context length, MATTR window, input column names, and local runtime options are in `config/pipeline.yaml`. The installed package carries an identical default configuration, so this also works:

```bash
uv run tide show-config
```

`config/paper.yaml` freezes the exact canonical manuscript run and requires the
model snapshots to exist locally. Its model revisions, probability definition,
batch size, and memory bounds are fixed; `config/pipeline.yaml` uses the same
numerical settings but permits the initial model download.

For the paper configuration, speakers are serialized as neutral labels in order
of first appearance (`S01`, `S02`, and so on). Each target turn is scored after
its speaker prefix on the preceding line, for example `S01:\n`, and all
preceding turns retain the same labels. The backend tokenizes the serialized
context and target jointly, then locates the target tokens from character
offsets before truncating the context. This preserves the model tokenizer's
canonical token sequence at the context-target boundary. The serialization
keeps the conditional-probability definition identical across dialogue types
and prevents source-specific labels such as `student`, `opponent`, `A`, or `B`
from entering the language-model readout. Set
`runtime.normalize_speakers: false` only when role labels are intentionally part
of the measurement design.

Lexical metrics use a separate documented preprocessing path: Unicode NFKC
normalization, collapsed whitespace, jieba 0.42.1 segmentation, Unicode
casefolding, and removal of tokens containing no letter or number. BGE inputs
retain the original turn text. Turns exceeding BGE's 512-token limit are split
into consecutive non-overlapping 510-content-token chunks. Each chunk is
embedded and L2-normalized; TIDE then computes a content-token-weighted mean
and normalizes the resulting full-turn vector. Set
`embedding.long_text_strategy: truncate` only to run the explicit
first-510-content-token sensitivity specification.

Custom serialization templates must also preserve a tokenizer boundary between
the speaker prefix and target text. TIDE raises an explicit error if a token
spans that boundary. It likewise rejects a request before inference when the
untruncated target would exceed either the pinned model's position limit or the
configured language-token budget; target turns are never silently truncated.
To bound memory without changing the conditional-probability definition, the
backend runs the causal transformer's hidden-state computation once per batch
and projects only target positions to the vocabulary. Target logits are
processed in fixed chunks controlled by `runtime.language_logit_chunk_size`
(default: 64), rather than materializing context-position logits that are never
used.

## Reproduce the validation analyses and data-driven figures

The public numerical table contains 1,567 turn rows from 103 dialogues and no
dialogue text. It also contains the text-free outputs of the documented BGE
truncation sensitivity. To regenerate the cluster-aware validation report, the
two data-driven paper figures, and an additional diagnostic heatmap:

```bash
uv run --extra analysis tide reproduce \
  data/paper/deidentified_turn_metrics_ratings.csv \
  --output-dir outputs/paper \
  --bootstrap-replicates 5000 \
  --cv-repeats 5
```

Figure 2 shows the repetition, lexical-redistribution, and context-conditioning
checks computed from the synthetic diagnostic cases. Figure 3 shows paired
held-out gains over structural length and established text descriptors. The
additional heatmap is retained for inspection but is not a main-text figure.
All generated figures use the manuscript's STIX serif family and embedded
TrueType fonts. Figure 1 is the only conceptual diagram.

Speaker-relative standardization groups turns by both dialogue and speaker.
Trajectory features are constructed separately for each raw readout from the
observed turn sequence; scaling for prediction is learned only within each
training fold. The validation report treats dialogue as the dependence and
hold-out unit. It uses dialogue-clustered uncertainty, nonlinear length
controls, dialogue-bootstrap confidence intervals, false-discovery-rate
adjustment, and nested grouped cross-validation.

To run only this validation report:

```bash
uv run --extra analysis tide validate \
  data/paper/deidentified_turn_metrics_ratings.csv \
  --output outputs/paper/brm_validation_report.md \
  --tables-dir outputs/paper/validation_tables \
  --bootstrap-replicates 5000 \
  --cv-repeats 5
```

This analysis treats dialogue as the dependence and hold-out unit. It uses
dialogue-clustered uncertainty, nonlinear length controls, dialogue bootstrap
confidence intervals, false-discovery-rate adjustment, and nested grouped
cross-validation. Baseline families are named for the rival explanation they
test: nonlinear length, conventional lexical/semantic descriptors,
information readouts, and the full TIDE family.
Every Markdown table is also written as a separate CSV so reported results can
be checked without parsing prose.

## Run the constructed-input diagnostics

The diagnostic cases hold context or target wording fixed while changing one
defined property. Compute their metrics with the same pinned model instruments,
then generate a report covering all nine readouts:

```bash
uv run --extra models tide compute data/demo/diagnostic_cases.csv \
  --config config/pipeline.yaml \
  --output outputs/diagnostic_metrics.csv

uv run tide diagnose outputs/diagnostic_metrics.csv \
  --output outputs/diagnostic_report.md
```

`data/demo/diagnostic_metrics.csv` records the frozen output used for Figure 2;
the commands above reproduce it from the synthetic text.

The report verifies deterministic boundary behavior and shows model-instrument
responses to repetition, paraphrase, grounded updating, topic detachment,
lexical redistribution, and context removal. These are computational checks,
not psychological-validity claims.

## Human-rating rubrics

`docs/turn_level_creative_thinking_rubric.md` contains the complete turn-level adaptation of TTCT Fluency, Flexibility, and Originality for argumentative dialogue. Dimension scores should be retained separately; the composite does not replace dimension-level analysis.

`docs/turn_level_critical_thinking_rubric.md` contains the study's turn-level
adaptation of the CTAR Analysis, Evaluation, and Inference dimensions. The
study data label the third dimension Reasoning. Paired rater values for both
rubrics are retained in the de-identified numerical table.

## Test and quality checks

The tests do not download either model. Pure metric functions and pipeline orchestration use constructed values and a deterministic fake backend.

```bash
uv sync --extra analysis
uv run ruff check .
uv run pyright
uv run pytest --cov=tide --cov-report=term-missing
uv build
```

## Data availability and privacy

Raw dialogue transcripts are not included because of privacy restrictions and
are not required for statistical reproduction. The public paper table contains
only coded identifiers, pseudonymous within-dialogue speaker labels, numerical
metrics, and numerical ratings. The synthetic file is sufficient for testing
the text-to-metrics pipeline. See `data/README.md` and `DATA_LICENSE.md`.

## Citation

Please use the metadata in `CITATION.cff`. GitHub's **Cite this repository** menu can export BibTeX and APA-formatted software citations.

## License

Source code is released under the MIT License. The synthetic data, de-identified numerical table, and rubric are licensed under CC BY 4.0; see `DATA_LICENSE.md`.
