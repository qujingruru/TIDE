# Data

This repository contains no raw participant dialogue.

- `demo/chinese_debates.csv` is a fully synthetic Chinese debate dataset written for this software demonstration. It contains no participant data.
- `paper/deidentified_turn_metrics_ratings.csv` contains only coded dialogue/turn identifiers, numerical metrics, numerical human ratings, dialogue condition, and a within-dialogue pseudonymous speaker identifier. Dialogue text and direct identifiers are absent.

The paper table is sufficient to verify the primary pooled correlations and
regressions summarized in TIDE's validation report, together with the
data-driven Figure 2, without exposing student speech. Raw dialogue text is
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

## Paper table fields

The rating fields `CR01_avg`, `CR02_avg`, and `CR03_avg` are turn-level fluency, flexibility, and originality ratings. `cr_turn_composite` is their sum. The `ct_*` fields contain the critical-thinking ratings. Missing creative-thinking values indicate turns that were not assessable; they are not zero scores.
