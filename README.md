# TIDE

**TIDE (Turn-level Information-theoretic Dialogue Evaluation)** is an open, local pipeline for measuring turn-level properties of dialogue. It uses language models only as read-out instruments: Qwen2.5-1.5B supplies next-token probabilities and BGE-base-zh-v1.5 supplies sentence embeddings. All metric calculation and statistical analysis are deterministic code. TIDE does not call a commercial or generative judging API.

TIDE accompanies the manuscript *What Can Information-Theoretic Metrics Tell Us about Student Thinking in Dialogue? A Design Framework, Emerging Boundary Evidence, and an Open Pipeline*.

## What is included

- A pip-installable Python package and `tide` command-line interface.
- Six turn-level metrics: lexical entropy, MATTR, mean surprisal, maximum sentence-level surprisal, semantic distance to the preceding partner turn, and semantic distance to the speaker's previous turn.
- Immutable Hugging Face model revisions and all parameters in one YAML configuration.
- A fully synthetic Chinese debate dataset for end-to-end use.
- Unit tests with constructed inputs of known answers.
- The turn-level creative-thinking rubric used in the study.
- A de-identified numerical table with metrics and ratings but no dialogue text.
- Reproduction code for the validation report and data-driven paper figures. The authored HTML source and author-approved screenshot for the conceptual Figure 3 are included under `paper_assets/`.

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

The output deliberately omits dialogue text. Each row contains identifiers, turn length, and the six metrics.

```text
dialogue_id, turn_id, turn, speaker, n_chars,
lexical_entropy, mattr, surprisal_mean, surprisal_sent_max,
sem_dist_partner, sem_dist_self
```

The first semantic distance in a dialogue and the first same-speaker distance are undefined and are written as missing values.

## Configuration

All model identifiers, immutable revisions, the random seed, context length, MATTR window, input column names, and local runtime options are in `config/pipeline.yaml`. The installed package carries an identical default configuration, so this also works:

```bash
uv run tide show-config
```

## Reproduce the paper analyses and figures

The public numerical table contains 1,567 turn rows from 103 dialogues and no dialogue text. To generate the validation report and Figures 1 and 2:

```bash
uv run --extra analysis tide reproduce \
  data/paper/deidentified_turn_metrics_ratings.csv \
  --output-dir outputs/paper
```

Figure 1 is recalculated from partial correlations that control turn length. Figure 2 is recalculated from semantic-distance quintiles with one-standard-error bars. Both use the manuscript's STIX serif family and embedded TrueType fonts. Figure 3 is a conceptual design artifact rather than a statistical plot; its editable HTML, approved screenshot, and rendered PDF are preserved in `paper_assets/`.

The hierarchical-regression implementation uses a correctly signed nested-model F test. The original exploratory script had reversed the residual-sum-of-squares subtraction; tests now protect the corrected calculation.

## Creative-thinking rubric

`docs/turn_level_creative_thinking_rubric.md` contains the complete turn-level adaptation of TTCT Fluency, Flexibility, and Originality for argumentative dialogue. Dimension scores should be retained separately; the composite does not replace dimension-level analysis.

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

Raw dialogue transcripts are not included because of privacy restrictions. The public paper table contains only coded identifiers, pseudonymous within-dialogue speaker labels, numerical metrics, and numerical ratings. See `data/README.md` and `DATA_LICENSE.md`.

## Citation

Please use the metadata in `CITATION.cff`. GitHub's **Cite this repository** menu can export BibTeX and APA-formatted software citations.

## License

Source code is released under the MIT License. The synthetic data, de-identified numerical table, and rubric are licensed under CC BY 4.0; see `DATA_LICENSE.md`.
