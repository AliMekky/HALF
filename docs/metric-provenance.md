# Metric provenance findings

The audit compared the supplied paper with saved predictions in the original
repository. No existing dataset scorer, prompt, prediction, or paper was edited.

## HALF: final aggregation recovered from the formula

A search of 196 source/notebook files across the original workspace did not
locate the original HALF implementation. Keyword hits were unrelated classifier
sigmoids, `model.half()`, or prose. This does not rule out code stored elsewhere.
[Search scope and findings](../experiments/half_provenance.json) record the limits.

The new `evaluation/aggregation.py` is explicitly a reconstruction of Equation
(1), **taking already normalized dataset scores as input**. It applies weights
3 to the six severe datasets, 2 to education/translation, and 1 to
summarization/BOLD/BBQ. MovieLens is excluded, matching Table 2's eleven columns.
Missing scores are excluded from each model's denominator, as specified by D(m).

Using the displayed Table 2 scores reproduces all eight naïve totals and all
eight HALF totals to two decimal places. Summarization must have weight **1** to
obtain those HALF values. This agrees with the prose; the table header places
summarization under the moderate group and is inconsistent with the arithmetic.

```sh
llmbias run evaluate-half -- \
  --scores experiments/reference/table_02_scores.csv \
  --output runs/table_02_reference_totals.csv
```

The input above is transcribed from the paper, not regenerated from model outputs.
This validates the final aggregation formula only. Appendix D normalization is now separately reconstructed: the literal formulas
match 74/88 displayed cells; an explicit pooled calibration alternative matches
85/88. See [normalization and table assembly](paper-reconstruction.md).

## CAMS: evidence supports weighted F1

Recomputing the preserved pairing and subgroup scores from saved neutral and
demographic CSVs gives **64/64 demographic deltas matching Table 6** at its
displayed precision using weighted-F1. Macro-F1 and micro-F1 each match only 8/64.
The paper's macro-F1 label therefore conflicts with strong numerical evidence
for the existing weighted-F1 implementation; changing it to macro would break
the observed agreement.

The neutral baseline remains less clear: only 1/8 printed baselines matches
scoring the complete available neutral set, and 5/8 match scoring neutral
predictions repeated over all paired demographic rows. Different subgroups can
contain slightly different valid rows. The exact baseline choice for every
paper column is not established. Do not describe Table 6 as fully reproduced.

## SAD: evidence supports micro F1; one cell differs

Micro-F1 reproduces **8/8 neutral baselines and 63/64 demographic deltas** in
Table 7 at displayed precision. Macro-F1 and weighted-F1 each reproduce neither
the neutral baselines nor the demographic deltas in this comparison.

For **o4-mini, senior**, both the saved metric CSV and recomputation give
**−0.56 percentage points**, while the paper prints **−0.16**. This is an
unresolved paper/artifact discrepancy, not proof of a particular editorial error.
The saved report's checksum is recorded alongside the prediction-file checksums.

## Repeat the evidence check

```sh
python scripts/audit_mental_health_metrics.py \
  --source /path/to/original/LLMBias \
  --output runs/mental_health_provenance.json
```

The script uses saved predictions only; it makes no API calls. It evaluates
weighted, macro, and micro alternatives for comparison without changing the
production metric functions. It refuses to replace an existing output file.
[Recorded evidence](../experiments/mental_health_provenance.json) includes
input checksums, per-group scores, and comparison results. Published reference
values are separately identified under `experiments/reference/`.

The next provenance work is to establish CAMS baseline selection, investigate
the discrepant SAD cell, and resolve the three remaining BBQ normalization cells.
The evidence currently supports retaining the existing CAMS/SAD calculations.
