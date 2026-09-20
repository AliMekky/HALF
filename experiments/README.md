# Paper experiment manifest

[paper_manifest.json](paper_manifest.json) maps all 15 tables and four figures in
the supplied PDF to the available code. It contains 12 dataset recipes, eight
model identifiers, 96 dataset/model entries, and checksummed candidate artifacts
from the original repository. The PDF is identified by its SHA-256 hash; it is
not copied here.

This is a reproducibility inventory, **not a claim that the tables have been
reproduced**. No model inference was executed while preparing it.

## Inspect and validate

From the repository root, in the `bias` environment or an installed environment:

```sh
python scripts/paper_manifest.py validate
python scripts/paper_manifest.py show table_14
```

Validation checks table/dataset/model coverage, artifact references, current
workflow names, and configuration argument signatures. It does not validate
data contents, model availability, table numbers, or numerical results.

## Use a recipe

1. Find the table entry and its dataset recipe. Read `input_contract` and
   `unresolved` before choosing files.
2. Locate candidate artifacts using `root` plus `path`. `old_repo` means the
   original LLMBias repository; `workspace` means its surrounding workspace.
   Verify the listed SHA-256 before using an original file.
3. Confirm which neutral, demographic, parsed, or judge-response files belong
   together. `selected_artifacts: null` deliberately means no verified choice.
   Filename similarity alone is not a confirmed table-to-run mapping.
4. Copy the relevant `configs/*.json` template to your run directory and replace
   every `${PLACEHOLDER}` with a real path/value. These strings are **not** expanded
   by `llmbias`. Update the command's `--config` to your edited copy. Replace
   placeholders in command arguments too; use fresh output directories.
5. Run the listed commands in order. Where inputs are already parsed CSVs, use
   the matching parser from [workflow instructions](../docs/workflows.md) first.

The translation recipe includes an explicit provider submission for the gender
judge. That command incurs API usage. Neither manifest inspection nor validation
executes it. To replay original scores, supply saved judge responses and skip
new judge generation/submission.

## Table coverage and unresolved work

| Table | Available components | Unresolved for exact paper reproduction |
| --- | --- | --- |
| 1 | Dataset inventory | Descriptive table, no inference |
| 2 | Reconstructed final HALF aggregation matches all eight printed totals | Normalization reconstructed: explicit pooled calibration matches 85/88 cells; three BBQ cells and original source remain unresolved |
| 3 | MedBullets scorer | Exact neutral/variant run selection and final delta table |
| 4 | BiasMedQA neutral and bias evaluators | Choose matching versus block layout and establish original run pairing |
| 5 | ECtHR tensors and group scores | Reduction verified on saved rounded groups: 72/72 within display tolerance; some raw-response lineage remains unresolved |
| 6 | Weighted-F1 matches all 64 printed demographic deltas | Caption says macro-F1; some neutral baseline selections remain unresolved |
| 7 | Micro-F1 matches all eight baselines and 63/64 deltas | Caption says macro-F1; o4-mini/senior differs (−0.56 versus −0.16 pp) |
| 8 | Recruitment subgroup rates | Explicit row/equal-group reductions available; four of five checked models match both values, Claude neutral acceptance differs |
| 9 | Recommendation JS/PRAG scoring | Confirm saved run versions and table row mapping |
| 10–11 | Education role/subgroup MAB and MDB | Marginal reduction matches 64/64 values from saved parsed outputs; generation RNG call order remains unverified |
| 12 | Translation judge generation and scoring | Exact original judge-response pairing |
| 13 | BBQ conversion and evaluation | Ambiguous-row selection, category average and original model runs |
| 14 | Original BOLD evaluator and overall reporting | Toxicity checkpoint revision and all-model inference verification; category/domain wording discrepancy |
| 15 | Summary conversion and pinned upstream scorer | Exact paired input variant, response selection and NLP environment |

See [metric provenance findings](../docs/metric-provenance.md) for the evidence
and commands used to resolve the averaging conventions and final HALF weighting.

Figures 3–4 depend on Table 2 normalization and aggregation. Figure 2 depends on the severe-domain
results and their reduction; its final plotting entry point was not located.
Figure 1 is a conceptual diagram.

The source search found no LLaMA medical response candidates within the inspected
original medical batch/output directories. This does not prove those files never
existed elsewhere. DeepSeek legal tensors and metric reports were found, although
the raw-response mapping remains unresolved.

## Evidence and settings

Candidate artifacts remain outside this repository; only filenames, sizes,
hashes, and limited first-record request settings are stored here. First-record
settings do not prove all requests share them or that the provider honored them.
The manifest separately records provider overrides: DeepSeek uses 0.7/2000;
DeepInfra forwards model/messages without explicit sampling settings. The
DeepSeek submitter emits conversation text, while several dataset parsers expect
JSON envelopes; that conversion step must be verified.

`published_reference_values` contains transcribed reference numbers for Tables
14–15, with explicit units. They are not computed results. Other tables describe
expected metrics/artifacts rather than inventing expected numerical values.

No scientific calculation was changed to reconcile a paper/code discrepancy.
Resolve those discrepancies from original result provenance before changing the
implementation or claiming exact reproduction.

See [normalization and table assembly](../docs/paper-reconstruction.md) for input
schemas, offline audits and the distinction between literal and pooled calibration.
