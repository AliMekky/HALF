# Normalization and paper-table assembly

These modules reconstruct missing reporting steps from the supplied paper and
saved outputs. They do not replace the original dataset scorers or change
published predictions. They are not evidence of a complete rerun of the paper.

## Normalize domain metrics

`llmbias run normalize-paper` accepts a CSV with columns
`model,dataset,metric,value`. Supply an explicit model cohort: every metric must
contain exactly that cohort. Normalization uses population standard deviation
and the Appendix D sigmoid direction, with inverse sigmoid for bias measures.
BOLD sentiment also uses the printed inverse direction. MovieLens uses the
positive direction and is excluded from the eleven-dataset HALF score.

| Dataset | Required metric names |
| --- | --- |
| medbullets, medical_bias, CAMS, SAD | `mean_abs_delta` |
| ecthr | `mean_gd` |
| djinni | `flip_rate` |
| education_ranking | `mean_teacher_mab` |
| mt_gender | `mean_language_bias` |
| ontonotes | `word_list`, `inclusion`, `hallucination` |
| bold | `sentiment`, `toxicity` |
| bbq | `acc_bias` |
| movielens | `mean_recency_js` |

Use consistent units across models for each metric. BOLD and summarization
average their normalized components. Legal follows the displayed equation:
normalize the mean GD across attributes. Zero variance raises an error unless
`--zero-variance neutral` is explicitly selected; the paper does not define this
case. BBQ requires `--bbq-mode signed` or `absolute` because this choice is not
resolved by the published description.

Using model-level values transcribed from Tables 3–15, literal normalization
matches **74/88** displayed Table 2 cells. An explicit alternative fits translation
and BBQ calibration across model/language or model/category pairs, then applies
it to model means. With absolute BBQ bias this matches **85/88** cells. This is
an investigated interpretation, not a recovered original implementation.
Three BBQ scores remain different: GPT-4.1-mini, o4-mini and DeepSeek-V3.

Run the pooled reference example from the repository root after installation:

```sh
llmbias run normalize-paper -- \
  --metrics experiments/reference/pooled_metrics.csv \
  --calibration experiments/reference/pooled_calibration.csv \
  --models claude4 gpt41 gpt41mini o4mini deepseek_v3 llama1b llama3b llama8b \
  --bbq-mode absolute --output-dir runs/normalized-reference
llmbias run evaluate-half -- \
  --scores runs/normalized-reference/half_inputs.csv \
  --output runs/reconstructed-half.csv
```

Outputs include long scores, wide HALF inputs and a JSON record of fitted
parameters and policies. Choose a fresh output directory. Calibration CSVs use
`model,dataset,metric,group,value`; only translation and BBQ support this
alternative. Their group means must agree with the supplied model-level metrics.
Without `--calibration`, calibration uses the model-level values directly.

These reference inputs are rounded published numbers, not regenerated predictions.
Consequently this example does not reproduce every Table 2 cell or guarantee its
final totals. Independently, aggregation of the actual transcribed Table 2 scores
matches all eight published naïve and HALF totals. Summarization has weight 1;
see [metric provenance](metric-provenance.md).

## Assemble domain tables

```sh
llmbias run assemble-table -- legal --input runs/legal-groups.csv --output runs/table-05.csv
llmbias run assemble-table -- education --input runs/education-cells.csv --output runs/education-tables.csv
llmbias run assemble-table -- recruitment --input runs/recruitment-groups.csv \
  --neutral runs/neutral-decisions.csv --weighting row --output runs/table-08.csv
llmbias run assemble-table -- deltas --input runs/deltas.csv --output runs/delta-table.csv
```

Input schemas and reductions:

- **Legal:** `model,attribute,group,mf1`, with percentage mF1 and attributes
  `state`, `gender`, `age`. Computes equal-weight group mean, population standard
  deviation (GD), and worst group. Saved rounded group scores agree with **72/72**
  Table 5 values within half a displayed unit (0.05 percentage points). Scores
  recomputed from unrounded tensors agree with 62/72; rounding order matters.
  The function preserves its input precision rather than silently rounding.
- **Education:** `model,role,gender,ethnicity,mean_score`, with a complete 2×3 grid
  for each model/role and A–E scores mapped to 1–5. Standardizes the six cell
  means, averages by demographic dimension, then calculates MAB/MDB. Saved parsed
  outputs reproduce **64/64** values in Tables 10–11 within display tolerance.
- **Recruitment:** group CSV uses `model,group_type,group_key,flip_rate,count`,
  with fractional flip rates; neutral CSV uses `model,index,decision` with unique
  inputs and Admit/Reject decisions. Uses only the `gender_ethnicity` partition
  to avoid counting overlapping summaries. Explicit `row` or `equal_group`
  weighting is required. Both policies match the displayed flip rates for five
  checked models, so historical weighting is not established. Four models also
  match neutral acceptance; Claude gives **26.2% versus the paper's 26.3%**.
  The three LLaMA parsed recruitment CSVs were not located for this audit.
- **Deltas:** `model,variant,delta_pp`, already verified percentage-point deltas.
  This command pivots values; it does not choose baselines or infer run pairings.

## Repeat the offline checks

```sh
python scripts/audit_normalization.py --output runs/normalization-audit.json
python scripts/audit_table_assembly.py \
  --source /path/to/original/LLMBias --output runs/table-assembly-audit.json
```

Neither script submits requests. The first needs only checked-in reference
values; the second requires the original saved CSVs/tensors and records their
checksums. Choose fresh output files. Recorded results:
[normalization](../experiments/normalization_validation.json) and
[table assembly](../experiments/table_assembly_validation.json).

Remaining provenance gaps include the three BBQ normalization cells, Claude's
neutral recruitment value, CAMS baseline selection, and the SAD o4-mini senior
cell. Exact response/judge pairing and some provider conversions also remain in
the [experiment manifest](../experiments/README.md). No correction constants have
been introduced to force agreement with the paper.
