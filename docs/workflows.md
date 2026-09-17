# Notebook workflow guide

The modular entry points expose retained notebook implementations and the explicitly documented BOLD reconstruction with explicit inputs and outputs. They preserve historical prompts, response formats, pairing, filtering, random-state behavior, formulas, and alternate versions. They do not establish which notebook execution produced a published table.

## Running a workflow

Run from the repository root after installation, or prefix commands with `PYTHONPATH=src python -m llmbias.cli` in place of `llmbias`.

```sh
llmbias list
llmbias coverage
llmbias workflow evaluate-translation
llmbias workflow evaluate-translation --config examples/translation-evaluation.json
llmbias workflow prepare-movielens --config examples/movielens-preparation.json --output runs/movielens.csv
```

Edit example paths to refer to your saved inputs. Configuration files contain keyword arguments for the selected entry point, listed below. Paths are relative to your working directory. DataFrame parameters accept CSV, Parquet, JSON or JSONL file paths; `frames` in education preparation is a list of paths. Other list/dictionary parameters retain their historical structure and are supplied directly as JSON values (including `batches`, `data`, and `gt_answers` for qualitative enrichment).

Functions that return a DataFrame or metrics dictionary can export it through `--output` (CSV for a DataFrame, JSON otherwise). Functions with output-path arguments write their historical output formats themselves; omit `--output` for those. The workflow dispatcher checks output paths, including defaults, and rejects existing files or nonempty output directories. Direct Python functions and older `run` scripts retain their original overwrite behavior. Run on copied inputs: some preparation functions mutate their input DataFrame.

## Coverage and versions

- CAMS and SAD have separate OpenAI, Anthropic and DeepSeek parsers. SAD retains both output modes and original label order. Recruitment has OpenAI and Anthropic decision parsers; no dedicated DeepSeek decision parser was found.
- Medical bias has neutral, seven-block (1,273 responses each), cleaned-answer, and prompt-matching evaluators. They are distinct historical workflows. Medical neutral requests retain their separate prompt and temperature 0.0.
- Recommendation cell 382 and earlier cell 379 remain separately selectable. Recruitment cell 304 and cell 306 also remain separate.
- Translation judge generation preserves cell 452's model and temperature, with scoring from cell 456. Earlier judge definitions from cell 450 remain available as a Python module.
- BBQ conversion retains the historical `gpt-4.1-2025-04-14` output key. The upstream scorer default is `o4-mini-2025-04-16`; pass the converter's key explicitly when evaluating its output (as in the example). Filename-based provider detection remains unchanged.
- Summarization conversion preserves cell 412. Evaluation calls the preserved upstream `summary_bias` checkout, optionally selected through `upstream_root` and a separate `python` executable. Install that checkout's requirements, spaCy model and NLTK resources in its environment. Upstream execution can download resources. The adapter command construction is tested; the full NLP pipeline has not been run during this migration.
- Active datasets now follow [paper scope](paper-scope.md). Out-of-scope tasks are removed from the package. [BOLD conversion and scoring](bold.md) use original-source implementations where available, with explicit provenance for unresolved historical choices.

Preparation recipes accept locally supplied source tables/files; they do not automatically download datasets. Source loading choices, manual notebook row edits, exploratory displays, and intermediate exports remain documented in the archive. The processed frozen datasets remain the inputs for replaying historical experiments. Sampling recipes retain assumptions about population sizes; some require substantial data. Education retains its stateful seeded RNG; recruitment retains its original random sampling.

Plotting requires the `analysis` extra; iterative multilabel sampling and Parquet input require the `datasets` extra. The CAMS intersectional adapter temporarily changes working directory to relocate all report artifacts; run it serially. Legal preprocessing includes gender labeling and placeholder detection, not a newly invented ethnicity transformation.

## Entry points and arguments

Parameter names match the Python functions. The source column uses zero-based notebook cell indices.

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
| `evaluate-recruitment-cell304` | `model, df_neutral, df_modified` | cell 304 |
| `evaluate-education` | `csv_dir_paths, output_dir, task='ranking', bootstrap=5000` | cell 367 |
| `evaluate-recommendation` | `data_dir, output_dir` | cell 382 |
| `evaluate-recommendation-cell379` | `data_dir` | cell 379 |
| `prepare-translation-judge` | `gold_csv, translations_jsonl, output_path, model_name=JUDGE_MODEL` | cells 451,452 |
| `evaluate-translation` | `input_path, gold_csv, output_path` | cell 456 |
| `convert-summarization` | `input_path, dataset_path, output_path` | cell 412 |
| `evaluate-summarization` | `input_path, output_dir, upstream_root=None, python=None` | cell 414 and preserved summary_bias checkout |
| `convert-bold` | `input_path, dataset_path, output_path, model_name, text_mode, anonymization, entities_path=None, require_complete=True, metadata_overrides_path=None, id_policy='exact', batch_path=None` | BOLD dataset metadata and saved provider envelopes |
| `evaluate-bold-sentiment` | `input_path, output_dir, expected_domains=None` | Dhamala et al. 2021 §4.1,A.2.4; LLMBias Appendix D.7 |
| `evaluate-bold` | `input_path, output_dir, toxicity_reduction, toxicity_provenance, toxicity_scores_path=None, checkpoint_path=None, label_order=None, threshold=None, batch_size=8, device='cpu', expected_domains=None` | Dhamala et al. 2021 §§3.3,4.1,4.2,A.2; LLMBias Appendix D.7 |
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

## Preservation checks

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m unittest discover -s tests -p test_notebook_workflows.py -v
```

The notebook checks use sanitized original-cell fixtures on temporary synthetic inputs and run without `legacy/`. They compare parser bytes, requests, table contents and aggregates against original source. Existing preservation tests additionally use the local artifact bundle. Optional plotting, iterative multilabel sampling, live provider APIs, and full upstream NLP execution have not been exercised end to end.

`docs/notebook-extractions.json` records source cells and their hashes. `scripts/extract_notebook_workflows.py` rebuilds generated modules from the preserved notebook without executing it. Handwritten dispatchers/adapters live outside that generator. The tests execute selected original source only on temporary fixtures; they never Run All the archived notebook.

Validation for this extraction: 31 tests passed with the local bundle; 22 notebook workflow tests passed from an isolated copy containing only `src/` and `tests/`, with no legacy artifacts. All 827 original-file hashes and 827 archived-copy hashes matched the migration manifest. Tests ran with the available Python 3.9.6 interpreter; the declared Python 3.10+ installation environment was not separately provisioned. Historical pandas groupby calls emit deprecation warnings and remain unchanged.
