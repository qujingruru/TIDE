# Turn-Level Creative Thinking Scoring Rubric for Dialogue

## 1. Purpose

This rubric adapts the three dimensions of the Torrance Test of Creative Thinking (TTCT)—**Fluency**, **Flexibility**, and **Originality**—for turn-level coding of argumentative dialogue.

The rubric is designed to identify when and how a participant demonstrates creative thinking during an interaction. It evaluates only the content produced in the target turn, while using the surrounding dialogue to determine whether an idea is new, repeated, responsive, or derived from another speaker.

This is a contextual adaptation of TTCT, not a new theoretical model. The three TTCT dimensions are retained, but their behavioral indicators are operationalized for dialogue rather than narrative writing.

## 2. Unit of Analysis

The primary unit of analysis is one **speaker turn**.

For each target turn, coders should also review:

1. the immediately preceding turn by the other speaker;
2. the target speaker's most recent previous turn; and
3. additional earlier turns only when needed to determine whether an idea has already appeared.

Context is used to establish novelty, repetition, perspective change, and idea ownership. Scores must reflect only what the target speaker contributes in the current turn.

## 3. Eligibility

Each turn must first be classified as either **eligible** or **not eligible** for creative-thinking scoring.

### 3.1 Eligible turns

A turn is eligible if it contains at least one substantive contribution to the topic, including:

- a claim, reason, explanation, or example;
- a counterargument or response to another speaker;
- a hypothesis, analogy, or counterfactual;
- a substantive question that introduces a new assumption, distinction, or line of inquiry;
- a reframing, synthesis, or extension of an earlier idea.

### 3.2 Ineligible turns

A turn should be coded as **N/A** if it contains no meaningful opportunity to demonstrate creative thinking, for example:

- greetings or leave-taking;
- procedural instructions;
- simple acknowledgements such as “okay” or “please continue”;
- requests that only manage the conversation;
- empty, corrupted, or unintelligible content.

An ineligible turn is not equivalent to a score of zero. **N/A** means that creative thinking cannot reasonably be assessed in that turn. A score of zero means that the turn was eligible but showed no evidence of the relevant dimension.

## 4. Idea Unit

An **idea unit** is the smallest segment that expresses an independently meaningful proposition, reason, example, hypothesis, distinction, or proposed solution.

Two statements should be coded as separate idea units when either could stand as a distinct contribution to the discussion. Rewording the same proposition, adding rhetorical emphasis, or repeating an earlier claim does not create a new idea unit.

An idea counts as new for Fluency only if it has not already been expressed by the same speaker in substantially the same form within the dialogue.

## 5. Turn-Level Scoring

Each eligible turn receives three scores from 0 to 3.

### 5.1 Fluency

**Definition:** The number of relevant, non-redundant idea units introduced in the current turn.

| Score | Anchor |
|---:|---|
| 0 | No new relevant idea unit. The turn only repeats, paraphrases, or affirms existing content. |
| 1 | One new relevant idea unit. |
| 2 | Two distinct new relevant idea units. |
| 3 | Three or more distinct new relevant idea units. |

Fluency measures idea production, not verbosity. Examples, elaborations, and reformulations should not be counted as additional ideas unless they introduce substantively different content.

### 5.2 Flexibility

**Definition:** The extent to which the turn introduces, changes, or integrates conceptual categories, perspectives, explanatory frames, or argument strategies.

| Score | Anchor |
|---:|---|
| 0 | No identifiable perspective development; the turn merely repeats an existing frame or provides content with no meaningful conceptual direction. |
| 1 | Develops one existing perspective, conceptual category, or argument strategy without a clear shift. |
| 2 | Introduces one genuinely different perspective, conceptual category, representation, or argument strategy. |
| 3 | Meaningfully compares, combines, or integrates two or more distinct perspectives, or substantially reframes the problem. |

Possible perspective changes include, but are not limited to, movement between scientific, philosophical, psychological, social, ethical, historical, practical, or personal frames. Coders should identify functional changes in perspective rather than count disciplinary keywords mechanically.

### 5.3 Originality

**Definition:** The statistical uncommonness and meaningful novelty of the idea or idea combination in relation to other responses to the same dialogue topic.

| Score | Anchor |
|---:|---|
| 0 | The content is copied, directly paraphrased, irrelevant, or does not contain an assessable idea. |
| 1 | The idea is common, expected, or strongly conventional for the topic. |
| 2 | The turn contains an uncommon extension, example, analogy, distinction, or combination of otherwise familiar ideas. |
| 3 | The turn contains a rare, relevant, and generative idea or combination that opens a substantively new line of thought. |

Originality must be evaluated relative to responses to the **same topic or prompt**. Unusual wording alone is insufficient. An idea should not receive a high Originality score merely because it is obscure, incoherent, irrelevant, or based on an obvious factual misunderstanding. A factual error does not automatically eliminate creativity, but an idea whose apparent novelty depends entirely on that error should not receive a score of 3.

## 6. Idea Ownership and Interaction Rules

Dialogue creativity is often responsive. Coders should apply the following rules consistently:

- **Direct repetition:** Repeating or closely paraphrasing another speaker's idea does not count as a new idea and normally receives Originality = 0.
- **Extension:** Adding a new implication, mechanism, example, or consequence may count toward Fluency and may receive Originality = 1 or 2.
- **Transformation:** Applying another speaker's idea to a new domain or conceptual frame may count toward Flexibility and Originality.
- **Synthesis:** Combining ideas from multiple earlier turns into a new and meaningful structure may receive high Flexibility and Originality scores.
- **Creative questions:** A question may be scored when it introduces a new assumption, counterfactual, distinction, or direction. A purely procedural question is N/A.
- **Self-repetition:** Repeating the speaker's own earlier idea does not increase Fluency, even if the wording changes.

## 7. Turn-Level Composite Score

For an eligible turn, calculate:

\[
\text{Turn Creative Thinking Score}_i = F_i + X_i + O_i
\]

where:

- \(F_i\) is Fluency, from 0 to 3;
- \(X_i\) is Flexibility, from 0 to 3; and
- \(O_i\) is Originality, from 0 to 3.

The turn-level composite therefore ranges from 0 to 9.

The three dimension scores should always be retained separately. The composite score must not replace dimension-level analysis.

## 8. Dialogue-Level Aggregation

To preserve the original TTCT reporting range, aggregate eligible turns for each participant within each dialogue as follows:

\[
F_{12} = 4 \times \operatorname{mean}(F_i)
\]

\[
X_{12} = 4 \times \operatorname{mean}(X_i)
\]

\[
O_{12} = 4 \times \operatorname{mean}(O_i)
\]

\[
\text{Dialogue Creative Thinking Score} = F_{12} + X_{12} + O_{12}
\]

This produces:

- Fluency: 0–12;
- Flexibility: 0–12;
- Originality: 0–12; and
- total creative-thinking score: 0–36.

Only eligible turns are included in the means. Raw sums across turns should not be used to compare participants unless all participants had the same number of turns and the same opportunity to respond.

The original high, moderate, and low TTCT cutoffs should be treated as provisional until their validity is examined in the dialogue corpus.

## 9. Recommended Data Fields

The following fields should be stored for each turn:

| Field | Description |
|---|---|
| `creativity_eligible` | `1` for eligible, `0` for ineligible |
| `creativity_fluency` | Turn-level Fluency score, 0–3 |
| `creativity_flexibility` | Turn-level Flexibility score, 0–3 |
| `creativity_originality` | Turn-level Originality score, 0–3 |
| `creativity_composite` | Sum of the three dimension scores, 0–9 |
| `creativity_note` | Brief evidence or rationale for difficult cases |
| `idea_units` | Optional count or short segmentation of new idea units |
| `originality_reference` | Optional topic-specific idea category used to judge commonness |

For ineligible turns, the three dimension scores and composite score should be stored as missing values rather than zeros.

## 10. Coding Procedure

Coders should follow the same sequence for every turn:

1. Read the target turn and the required context.
2. Determine whether the turn is eligible.
3. Segment the turn into idea units.
4. Remove repetitions and identify which ideas are new for the target speaker.
5. Assign Fluency based on the number of new relevant idea units.
6. Assign Flexibility based on perspective or strategy changes.
7. Assign Originality using the topic-specific reference set.
8. Record a short note when a score is ambiguous or depends heavily on context.
9. Calculate the turn-level composite score.

## 11. Topic-Specific Originality Reference

Before full coding, the research team should construct a reference list of common idea categories for each dialogue topic. The list should be developed from a pilot sample and revised during coder calibration.

The reference should distinguish:

- common or expected ideas;
- less common extensions and combinations; and
- rare but relevant and generative ideas.

Coders should remain blind to experimental condition or participant group when judging originality. The reference list should be applied consistently across all groups responding to the same topic.

## 12. Coder Calibration and Reliability

Before full-scale coding:

1. Select a stratified pilot sample containing different topics, turn lengths, dialogue stages, and turn functions.
2. Have at least two coders score the same turns independently.
3. Compare idea-unit segmentation and all three dimension scores.
4. Discuss disagreements and revise ambiguous decision rules.
5. Repeat calibration until agreement is acceptable.

Weighted Cohen's kappa may be used for the ordinal turn-level dimension scores. An intraclass correlation coefficient may be reported for aggregated continuous scores. Internal consistency among the three dimensions is not a substitute for inter-rater reliability.

## 13. Interpretation

Turn-level scores identify moments of creative performance; they should not be interpreted as stable measures of a participant's general creative ability.

For analysis, researchers may report:

- the mean creative-thinking score across eligible turns;
- the proportion of eligible turns showing moderate or high creativity;
- the maximum or peak turn score;
- changes in creativity across dialogue stages; and
- the dialogue-level 0–36 aggregate score.

Because turns are nested within participants and dialogues, inferential analyses should account for this nested structure rather than treat all turns as independent observations.

## 14. Brief Coding Examples

### Example A: Procedural turn

> “Please continue with your argument.”

Eligibility: N/A. The turn manages the interaction but contributes no substantive idea.

### Example B: One conventional idea

> “Time is subjective because people feel that it passes more quickly when they are happy.”

- Fluency = 1: one new relevant idea;
- Flexibility = 1: one psychological perspective;
- Originality = 1: a common argument for the topic;
- Composite = 3.

### Example C: Multiple perspectives and an uncommon synthesis

> “Psychological time varies with emotion, but physical clocks remain stable. This suggests that the debate may involve two different levels of time rather than a single subjective–objective opposition.”

- Fluency = 2: two distinct relevant ideas;
- Flexibility = 3: psychological and physical perspectives are compared and integrated;
- Originality = 2: the two-level synthesis is an uncommon extension;
- Composite = 7.

### Example D: Substantive creative question

> “If a machine could predict every future event but had no conscious experience, would it possess a concept of time or merely process ordered states?”

- Fluency = 1: one new counterfactual idea;
- Flexibility = 2: introduces a computational perspective;
- Originality = 2 or 3, depending on its rarity in the topic-specific corpus;
- Composite = 5 or 6.

## 15. Recommended Label

In research reports, the instrument should be described as:

> **A turn-level adaptation of the TTCT Fluency, Flexibility, and Originality rubric for argumentative dialogue.**

This label makes clear that the underlying TTCT dimensions are retained while the coding indicators and unit of analysis have been adapted to the dialogue context.
