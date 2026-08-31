# Data

This repository contains no raw participant dialogue.

- `demo/chinese_debates.csv` is a fully synthetic Chinese debate dataset written for this software demonstration. It contains no participant data.
- `demo/diagnostic_cases.csv` contains synthetic controlled contrasts for repetition, paraphrase, grounded updating, topic detachment, lexical redistribution, and context removal.
- `demo/diagnostic_metrics.csv` is the frozen TIDE output for those synthetic cases and supplies the values in paper Figure 2.
- `paper/deidentified_turn_metrics_ratings.csv` contains only coded dialogue/turn identifiers, numerical metrics, numerical human ratings, dialogue condition, and a within-dialogue pseudonymous speaker identifier. Dialogue text and direct identifiers are absent.
- `paper/human_rating_criterion_reliability.csv` is the machine-readable version
  of manuscript Table S1, with dialogue-bootstrap intervals for ICC(2,1) and
  ICC(2,2). Its readable counterpart is
  `docs/table_s1_human_rating_reliability.md`.
- `paper/source_quality_dialogues.csv` lists four pseudonymous dialogue IDs with
  dialogue-level quality flags used for the exclusion sensitivity analysis. It
  contains no dialogue text or direct identifiers. The primary analysis keeps
  the original rated turn units; the sensitivity analysis excludes complete
  flagged dialogues rather than re-segmenting selected turns.

The paper table is sufficient to verify the primary pooled correlations and
regressions summarized in TIDE's validation report, together with the two
data-driven paper figures, without exposing student speech. Raw dialogue text is
not required for this reproduction and should not be added to the repository.
Do not attempt to re-identify participants or combine the table with external
identity data.

## Input schema for the demo

| Column | Required | Description |
|---|---:|---|
| `turn` | yes | Order of the turn within a dialogue |
| `speaker` | yes | Speaker label used only to locate prior same-speaker turns |
| `text` | yes | Turn text |
| `dialogue_id` | no | Dialogue grouping; omission treats the file as one dialogue |
| `turn_id` | no | Stable turn identifier; TIDE creates one if omitted |

The computed file omits `text`. `n_chars` counts NFKC-normalized non-whitespace
characters. `n_words` uses NFKC normalization, collapsed whitespace, jieba
0.42.1 segmentation, Unicode casefolding, and exclusion of tokens containing
no letter or number. Both are surface baselines.
`self_novelty`, `delta_surprisal`, and `peak_break` are derived relative to the
same speaker's earlier turns and are missing when no prior self turn exists.

## Paper table fields

`source_quality_flag` identifies rows from a dialogue included in the
dialogue-level source-quality sensitivity analysis, and
`source_quality_issue` gives the text-free flag category. The primary results
retain these rows; the sensitivity analysis removes the complete flagged
dialogue.

`sem_dist_partner_truncate`, `sem_dist_self_truncate`, and
`self_novelty_truncate` are the corresponding text-free semantic readouts when
only the first BGE window is retained. They support the declared long-text
sensitivity without releasing dialogue text.

The rating fields `CR01_avg`, `CR02_avg`, and `CR03_avg` are turn-level fluency, flexibility, and originality ratings. `cr_turn_composite` is their sum. The paired `CR01_R1`/`CR01_R2`, `CR02_R1`/`CR02_R2`, and `CR03_R1`/`CR03_R2` fields preserve the two de-identified rater values used to reproduce creative-thinking inter-rater reliability. `ct_analysis`, `ct_evaluation`, and `ct_reasoning` contain the adjudicated critical-thinking dimensions; `CT01_R1`/`CT01_R2`, `CT02_R1`/`CT02_R2`, and `CT03_R1`/`CT03_R2` preserve their two de-identified rater values. Missing creative-thinking values indicate turns that were not assessable; they are not zero scores.
