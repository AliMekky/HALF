# Running paper workflows

The sequence is: prepare data → build requests → submit through a provider →
parse saved responses → evaluate → produce reports. Existing prompts, pairing,
filtering, random-state behavior, and metric formulas are preserved.

- `llmbias build`: generate requests from a dataset; no API calls.
- `llmbias run COMMAND -- ...`: provider submission and standalone parsers/scorers.
  Append `--help` after `--` to inspect the script's arguments.
- `llmbias workflow NAME --config CONFIG.json`: run a preparation, conversion,
  evaluation, or reporting function using named arguments from JSON.
- `llmbias list` and `llmbias coverage`: inspect commands and dataset coverage.

Omit `--config` to inspect a workflow. Edit paths in `examples/` to your local
inputs and saved responses. Relative paths are resolved from the working directory.
Use fresh output locations; the workflow wrapper rejects existing output files or
nonempty output directories. Standalone scripts retain their original I/O behavior.

DataFrame arguments accept CSV, Parquet, JSON or JSONL paths. `frames` accepts a
list of paths. Functions returning tables or dictionaries support `--output`
(CSV or JSON respectively); omit it when the function already writes its outputs.

```sh
llmbias workflow parse-cams-openai --config examples/cams-parsing.json
llmbias workflow evaluate-translation --config examples/translation-evaluation.json
```

Source cell numbers below identify where the scientific implementation originated;
the source notebook is not needed to execute these functions.

| Workflow | Configuration arguments | Source |
| --- | --- | --- |
| `parse-cams-openai` | `input_path, output_csv_path` | cell 328 |
| `parse-cams-anthropic` | `input_path, output_csv_path` | cell 330 |
| `parse-cams-deepseek` | `input_path, output_csv_path` | cell 332 |
| `parse-sad-openai` | `input_path, output_csv_path, output_format` | cell 349 |
| `parse-sad-anthropic` | `input_path, output_csv_path, output_format` | cell 351 |
| `parse-sad-deepseek` | `input_path, output_csv_path, output_format` | cell 356 |
| `parse-recruitment-openai` | `input_path, output_csv_path` | cell 302 |
| `parse-recruitment-anthropic` | `input_path, output_csv_path` | cell 300 |
| `parse-education` | `input_path, output_csv_path` | cell 364 |
| `parse-medical-answer` | `input_path, output_csv_path` | cell 297 |
| `prepare-medical-bias-neutral` | `input_path, output_path, model_name='o4-mini-2025-04-16'` | cell 265 |
| `evaluate-medical-bias-neutral` | `predictions_csv, gold_path` | cells 271,273 |
| `evaluate-medical-bias-blocks` | `predictions_csv, gold_path` | cells 275–280 |
| `evaluate-medical-bias-matching` | `responses_csv, batches_path, gold_path` | cells 437,442,443 |
| `evaluate-medical-bias-cleaned` | `all_prediction` | cell 287 |
| `evaluate-cams` | `gold_path, neutral_path, sensitive_path, out_path` | cell 338 |
| `evaluate-cams-groups` | `gold_csv, model_configs` | cell 343 |
| `report-cams-intersectional` | `neutral_csv, sensitive_csv, gold_csv, output_dir, save_heatmaps=True` | cell 341 |
| `evaluate-sad` | `gold_csv, neutral_csv, sensitive_csv, model_name, output_csv_path` | cell 358 |
| `evaluate-recruitment` | `neutral_csv, sensitive_csv, model_name` | cell 306 |
| `evaluate-education` | `csv_dir_paths, output_dir, task='ranking', bootstrap=5000` | cell 367 |
| `evaluate-recommendation` | `data_dir, output_dir` | cell 382 |
| `prepare-translation-judge` | `gold_csv, translations_jsonl, output_path, model_name=JUDGE_MODEL` | cells 451,452 |
| `evaluate-translation` | `input_path, gold_csv, output_path` | cell 456 |
| `convert-summarization` | `input_path, dataset_path, output_path` | cell 412 |
| `evaluate-summarization` | `input_path, output_dir, upstream_root=None, python=None` | cell 414 and preserved summary_bias checkout |
| `evaluate-bold` | `input_path, output_dir, response_format='openai', id_format='bold', model_revision=None, device=None` | workspace conv_ai/eval.py |
| `convert-bbq` | `input_path, gold_csv, output_path` | cell 420 |
| `evaluate-bbq` | `result_dir, metadata_file, output_dir, model_key='o4-mini-2025-04-16'` | preserved BBQ/analysis_scripts/BBQ_bias_score.py |
| `prepare-legal-gender` | `input_path, output_path` | cell 386 |
| `prepare-legal-placeholders` | `input_path, output_path` | cell 395 |
| `reformat-claude` | `input_path, output_path` | cell 448 |
| `concat-responses` | `input_paths, output_path` | cell 472 |
| `prepare-education` | `frames` | cells 100–104 |
| `prepare-bbq` | `df` | cells 188 |
| `prepare-recruitment` | `df_cv, df_jobs` | cells 153 |
| `prepare-movielens` | `ratings, movies` | cells 220,222,226 |
| `prepare-medical-bias-sample` | `merged_df` | cells 48 |
| `prepare-mental-multilabel` | `df` | cells 71 |
| `prepare-translation` | `pro_path, anti_path` | cells 111,114,115,117 |
| `prepare-neutralization-results` | `df, responses_path` | cells 322,323 |
| `prepare-legal-gender-results` | `csv_path, jsonl_path` | cells 387 |
| `prepare-ontonotes` | `input_path` | cells 403 |
| `prepare-medbullets` | `df` | cells 31,32,37 |
| `prepare-medbullets-sample` | `df` | cells 40 |
| `prepare-medical-prompts` | `base_dir` | cells 213,216 |
| `report-recruitment` | `input_path, output_dir` | cell 308 |
| `report-legal` | `csv_path` | cell 418 |
| `enrich-medbullets` | `outputs, batches, gt` | cell 429 |
| `enrich-deepseek` | `outputs, batches` | cell 433 |
| `enrich-flip-cases` | `detailed, data, gt_answers` | cells 495–497 |

## Execution details

BBQ conversion preserves its `gpt-4.1-2025-04-14` output key; pass the same
`model_key` to the evaluator. The supplied example does so. Some parsers retain
provider detection based on filenames. Recruitment includes OpenAI and Anthropic
decision parsers; no dedicated DeepSeek decision parser was recovered.
Medical bias workflows handle different original response layouts; choose the
one matching your inputs, rather than treating them as interchangeable.

Education retains its stateful seeded RNG; recruitment retains its original
sampling. CAMS intersectional reporting temporarily changes the working directory
and should run serially. Summarization requires the separate NLP resources in
[setup](code-release.md). BOLD uses the [original project scorer](bold.md).
