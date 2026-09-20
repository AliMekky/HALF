# Paper datasets

`processed/` contains the twelve paper input tables and seven required supporting
files (including BBQ metadata). [The manifest](datasets-manifest.json) records
source paths, destinations, sizes and SHA-256 hashes. Copies preserve file content.

## Paper inputs

Paths below are relative to `data/processed/`. Pass the key to `llmbias build
--dataset KEY --model MODEL --output-dir DIRECTORY`.

| Dataset | Key | File |
| --- | --- | --- |
| MedBullets | `medical_data/medbullets` | `medical_data/medbullets.csv` |
| BiasMedQA | `medical_data/medical_bias` | `medical_data/medical_bias.csv` |
| CAMS | `mental_health_data/CAMS` | `mental_health_data/CAMS.csv` |
| SAD | `mental_health_data/SAD` | `mental_health_data/SAD.csv` |
| Djinni | `admission_data/djinni` | `admission_data/djinni.csv` |
| Education | `education_data/education_ranking` | `education_data/education_ranking.csv` |
| MovieLens | `recommendation_system/movielens` | `recommendation_system/movielens.csv` |
| ECtHR | `legal_data/ecthr` | `legal_data/ecthr.csv` |
| WinoMT | `translation_data/mt_gender` | `translation_data/mt_gender.csv` |
| OntoNotes | `summarization_data/ontonotes` | `summarization_data/ontonotes.csv` |
| BBQ | `conv_ai/bbq` | `conv_ai/bbq.csv` |
| BOLD | `conv_ai/bold` | `conv_ai/bold.csv` |

The original `BOLD.csv` filename is normalized to `bold.csv` to match the builder
key; its contents are unchanged. Other filenames and domain prefixes are retained
for compatibility. Archived experiments are not restored to the active registry.

## Import and use

```sh
python scripts/import_datasets.py --source /path/to/LLMBias
```

The importer copies only the files listed in the manifest, verifies hashes, and
refuses to overwrite different content. It does not import old outputs or
out-of-scope datasets. The old repository is needed only for this optional import.

`llmbias build` defaults to `data/processed/`. Override it using `--data-root` or
`LLMBIAS_DATA_ROOT`. Evaluation configurations use explicit paths. The supporting
files include medical gold labels, translation PRO/ANTI inputs, summarization
input variants, and `conv_ai/bbq_additional_metadata.csv`.

These data are present locally but Git-ignored, so they do not accompany a public
clone. Dataset redistribution follows each source's access terms; OntoNotes in
particular requires separate access. The manifest and this guide are versioned.

## Obtain data without the original repository

Download the upstream inputs below, prepare CSVs with the matching columns in
[schemas.json](schemas.json), and save them at the paths in the table above.
The upstream releases and HALF's processed subsets are distinct; hashes in
`datasets-manifest.json` identify the latter. For a new experiment, preserve row
order between requests, gold labels and evaluation files.

| Dataset | Upstream access | Preparation entry point |
| --- | --- | --- |
| MedBullets | [ChallengeClinicalQA](https://github.com/HanjieChen/ChallengeClinicalQA), MedBullets data | `prepare-medbullets`, then `prepare-medbullets-sample` |
| BiasMedQA | [Cognitive-bias benchmark](https://github.com/carlwharris/cog-bias-med-LLMs), linked dataset download | Combine bias prompts as `bias_type,prompt`; `prepare-medical-bias-sample`; retain aligned gold JSONL |
| CAMS | [CAMS dataset](https://github.com/drmuskangarg/CAMS) | Map post text and causal labels; retain `neutralized_text` and stable row indices |
| SAD | [SAD_v1.zip](https://github.com/PervasiveWellbeingTech/Stress-Annotated-Dataset-SAD) | Export workbook text and stressor columns; retain `neutralized_text` and stable row indices |
| Djinni | [Dataset collection](https://huggingface.co/collections/lang-uk/djinni-recruitment-dataset) | Export CV/job tables; `prepare-recruitment` |
| Education | [Dataset authors and paper](https://arxiv.org/abs/2410.14012) | Obtain MATH, generated, generated-wired and WIRED `all.parquet` tables; `prepare-education` |
| MovieLens | [GroupLens downloads](https://grouplens.org/datasets/movielens/) | Supply ratings and movies tables to `prepare-movielens` |
| ECtHR | [FairLex](https://huggingface.co/datasets/coastalcph/fairlex) | Use the ECtHR split and `processing/preprocessing/legal_loading.py` and `legal_sampling.py` |
| WinoMT | [mt_gender](https://github.com/gabrielStanovsky/mt_gender), `data/aggregates/en.txt` and PRO/ANTI resources | `prepare-translation` accepts `pro_path,anti_path` |
| OntoNotes | [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19), then [SummaryBias construction](https://github.com/julmaxi/summary_bias) | Generate variants using the upstream instructions; `prepare-ontonotes` |
| BBQ | [BBQ](https://github.com/nyu-mll/BBQ), `data/` and `analysis_scripts/additional_metadata.csv` | Combine category JSONL tables; `prepare-bbq`; place metadata at `conv_ai/bbq_additional_metadata.csv` |
| BOLD | [BOLD](https://github.com/amazon-science/bold), `prompts/` | Flatten prompts into the BOLD CSV schema, retaining domain and category |

Inspect each workflow with `llmbias workflow NAME`. Configurations use function
argument names as JSON keys. For example:

```json
{
  "frames": ["downloads/MATH/all.parquet", "downloads/generated/all.parquet",
             "downloads/generated-wired/all.parquet", "downloads/WIRED/all.parquet"]
}
```

Save this as `runs/education-preparation.json`, then run:

```sh
llmbias workflow prepare-education --config runs/education-preparation.json \
  --output data/processed/education_data/education_ranking.csv
```

For translation, use `{"pro_path":"downloads/en_pro.txt","anti_path":"downloads/en_anti.txt"}`
as the configuration for `prepare-translation`, with `--output` set to
`data/processed/translation_data/mt_gender.csv`.

Upstream data remain governed by their respective access terms. OntoNotes requires
obtaining access from LDC. No dataset is relicensed by this project's MIT license.

For CAMS/SAD, `neutralized_text` is a derived field, not an upstream label.
The request builders use it as input. If starting from raw posts, apply the
neutralization template in `processing/preprocessing/neutralization.py` and
merge saved responses with `prepare-neutralization-results`; inspect that
workflow for its input schema. For paired evaluation, keep the same index
mapping across neutral and demographic requests.
