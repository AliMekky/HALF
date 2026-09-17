# Important implementations in the original notebook

Historical review preceding the notebook extraction. For the current implementation and remaining gaps, see [workflow coverage](workflows.md). The tables below record the starting gaps, not current package status.

Source: `repo/LLMBias/data_analysis.ipynb` in the original workspace. The new repository's archival counterpart is `legacy/data_analysis.ipynb`. All cell numbers below are **zero-based cell-array indices**, not execution counts. The notebook has 501 cells, including 456 code cells. Execution counts are not monotonic, and 28 code cells have no saved execution count. Saved execution metadata does not identify the exact code that produced a published table.

## Highest-priority missing implementations

| Workflow | Cells | Important behavior | Current package coverage / proposed extraction |
| --- | --- | --- | --- |
| Medical cognitive-bias neutral baseline | 265 | Builds requests from question/options gold JSONL; has its own system prompt and temperature **0.0** | Not extracted. Add `preparation/medical_bias_neutral.py`; do not substitute the generic builder's prompt or temperature |
| Medical cognitive-bias evaluation | 268–288; 436–443 | Earlier positional/block-based scoring and later prompt-to-gold matching, followed by accuracy per bias type | Not extracted as complete workflows. Preserve separately named versions; avoid silently choosing or merging them |
| Medical answer cleanup | 297 | `extract_final_choice` and its application to saved Claude output | Not extracted. Preserve the exact parsing expression and fallback behavior |
| Recruitment decision parsing | 300, 302 | Anthropic/OpenAI envelopes → index, gender, ethnicity, decision | Metrics from 306 are extracted, but these parsers are not. Add task-specific parsing modules |
| CAMS parsing | 328, 330, 332 | Provider-specific extraction of `OUTPUT:` labels and demographic columns; existing skips retained | Metrics from 338/343 are extracted, but parsers are missing. Add three named parser entry points |
| SAD parsing | 349, 351, 356 | Provider-specific label parsing, fixed label order, one-hot/indices modes, demographic columns | Scoring from 358 is extracted, but parsers are missing. Preserve each mode and its current handling of incomplete text |
| Education response parsing | 364 | Provider/path-dependent extraction of role, index, demographics, answer | Scoring from 367 is extracted, but parser is missing |
| Recommendation evaluation orchestration | 382 | Loads outputs by provider/model, associates neutral and sensitive lists by request index and query type, computes metrics, writes gender/ethnicity/age aggregates | Only helper functions are extracted. The loading, pairing, aggregation, and export statements must also be wrapped into an explicit workflow |
| Summarization conversion and scoring bridge | 412, 414 | Restores source article text/instructions and article/pair IDs, reads different provider response formats, writes the upstream evaluation format; documents parsing/scoring commands | The generic merge helper is not an equivalent replacement. Extract cell 412 and add an explicit adapter to the existing upstream evaluator |
| BBQ evaluation preparation | 420 | Decodes metadata, maps responses to `ans0`/`ans1`/`ans2`, writes benchmark-format JSONL | Not extracted. Add conversion and an adapter to the existing BBQ scorer, retaining historical output field conventions |
| Translation judge configuration and scoring | 452, 456, 464–465 | Cell 452 selects judge `gpt-4o-mini-2024-07-18` and temperature **0**; later cells parse judge results, join gold labels, compute accuracy, pro/anti accuracy, and their difference | Judge-request function from 451 exists, but its execution configuration and final scoring are not packaged |
| CAMS intersectional and qualitative reporting | 341 | Per-group metrics, gender×ethnicity / gender×age analysis, heatmap export | Distinct from extracted 338/343; not yet extracted |

Extracting `def` statements is insufficient for these workflows. Top-level statements often implement the actual data joins, scoring, output formatting, and experiment configuration.

## Preparation that should become explicit recipes

| Dataset / area | Cells | Preparation to preserve |
| --- | --- | --- |
| Medbullets | 23–44 | Source loading, duplicate handling, age/gender extraction, age grouping, alternate sampling paths, CSV exports |
| Additional medical datasets | 9–22, 45–59 | BiasMD proportional sampling, biased medical JSON collection, per-bias sampling, DiseaseBuster sampling |
| Medical cognitive-bias prompts | 213–219 | Read `bias_type/no-mitigation/test` text files; replace the literal `GPT model` with `assistant`; export prompts |
| Mental-health datasets | 61–99 | Alternative source/split loading, multilabel and class-based sampling, legacy exports; some short cells also manually edit rows |
| Mental-health neutralization results | 319–325 | Extract generated neutral text and attach it to the dataset; inspect explicit dataset/file choices rather than depending on the current `df` |
| Education ranking | 100–105 | Concatenate four Parquet sources, drop duplicates, export ranking input |
| Education/admission source data | 119–141 | Load cognitive-bias sources, select/sample data, extract structured student attributes |
| Recruitment | 142–160 | Load CV/job data, normalize fields, match keyword/language/experience, sample candidate-job pairs, export records |
| MovieLens | 220–227 | Sample **200 users**, retain **10** top-rated and **10** recent films, derive genre/year profiles from ratings ≥4, format and aggregate prompts |
| BBQ | 187–191 | Load 11 categories, keep **ambiguous contexts**, sample **91 per category** with random state 42, export 1,001 rows |
| Translation | 111–118 | Read pro/anti source files, annotate type, concatenate, export; alternate sampling cells also exist |
| ECtHR | 369–377, 386–395 | Test-split preparation, alternative demographic filtering, requests to infer unknown gender, result import, and protagonist-placeholder extraction |
| OntoNotes | 403 | Preserve text, serialized instructions, sample ID, article ID, and pair ID in CSV conversion |

These are available notebook recipes, not verified descriptions of every released dataset. For example, the MovieLens comments mention other sample sizes, but the active statements in cell 220 select 200 users and ten films per anchor strategy. Configuration should be taken from executable statements and verified against the saved artifacts.

The ECtHR heading “add ethnicity” is broader than the implemented cell 395: that cell creates requests to identify protagonist placeholders. It does not itself implement a complete ethnicity intervention. Cells 386/387 have no saved execution counts; that alone neither proves nor disproves their historical use.

## Distinct versions and ancillary work

- Recruitment evaluation exists in cells 304 and 306. The package currently uses 306. Keep their provenance separate.
- Recommendation evaluation exists in 379 and 382. The package currently extracts helpers from 382, but not its complete driver. Do not present 379 and 382 as interchangeable.
- Translation judge construction exists in 450 and 451. The package uses 451. Cell 452 contains important settings outside that function. Cells 456 and 465 have closely related scoring bodies but different execution/export settings.
- Medical cognitive-bias evaluation has both positional/block assumptions and prompt-matching logic. Determine the selected historical artifact chain before assigning a default paper workflow.
- Cells 308 and 341 contain reporting/plotting; 424–443 and 490–499 contain qualitative example assembly, scoring, comparisons, and enrichment. Only named functions from 490 were extracted previously.
- Cells 360/448 reformat provider files; 469–472 split/concatenate translation jobs. Retain them as explicit utilities, with original files preserved and output paths supplied by the caller.
- Dream-interpretation scraping in 205–210, synthetic demographic exploration in 163, and other alternate datasets should remain labeled exploratory until their study role is established.

## What I did not find as a complete notebook workflow

- A dedicated BOLD evaluation pipeline. Loading/inspection and generation support exist.
- A distinct education-admission evaluator. The section titled “Admission Evaluation” at 303 leads to **Djinni recruitment** evaluation, not a demonstrated `education_ga` scorer.
- A clearly separate dreaddit scoring workflow comparable to the CAMS and SAD sections. Preparation and neutralization handling exist; CAMS label parsing must not be assumed to apply unchanged to dreaddit's prompt format.

Do not fill these gaps by inventing metrics or repurposing another task's parser. Keep their coverage status explicit until a matching historical implementation is located.

## Recommended next structural pass

1. Extract task-specific parsers first; they connect saved raw outputs to the already-migrated metrics.
2. Extract complete scoring drivers for medical cognitive bias, MovieLens, translation, BBQ, and summarization. Move top-level processing into explicit functions without changing order, formulas, parsing, or filters.
3. Move historical settings into named configurations, including the medical neutral baseline and translation judge settings. Preserve multiple versions by source-cell provenance instead of silently consolidating them.
4. Add preparation recipes separately from saved-data replay. Manual edits, RNG state, external data, and API-generated labels can prevent a fresh preparation run from exactly recreating historical files; the frozen processed artifacts remain the reproduction inputs.
5. Add small offline fixtures per workflow and compare their parsed records, counts, intermediate tables, and final aggregates with the selected original cells. Perform extraction on temporary copies, never overwrite the preserved outputs.

No notebook cells were executed during this review. This document records implementation coverage and migration priorities, not a scientific re-evaluation of the accepted work.
