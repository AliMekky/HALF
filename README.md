# HALF

[Paper](https://arxiv.org/abs/2510.12217) · [Citation](CITATION.cff)

## Install

```sh
conda create -n bias python=3.10
conda activate bias
python -m pip install -c requirements/constraints.txt -e '.[providers,analysis,datasets,bold-model,summarization]'
```

If `bias` already exists, activate it directly. Export your provider's API key
using the variable names in [.env.example](.env.example).

## Data

Ten of the 12 datasets are included in `data/processed/`. MovieLens and OntoNotes
are not redistributed because of their licences. Rebuild them from the original
sources with `scripts/restore_data.py`, following [data/README.md](data/README.md),
which also lists each dataset's source and licence.

Use these dataset keys with `llmbias build --dataset`:

| Dataset | Key | Included |
| --- | --- | --- |
| MedBullets | `medical_data/medbullets` | Yes |
| BiasMedQA | `medical_data/medical_bias` | Yes |
| CAMS | `mental_health_data/CAMS` | Yes |
| SAD | `mental_health_data/SAD` | Yes |
| Djinni | `admission_data/djinni` | Yes |
| Education | `education_data/education_ranking` | Yes |
| MovieLens | `recommendation_system/movielens` | Rebuild |
| ECtHR | `legal_data/ecthr` | Yes (CC BY-NC-SA 4.0) |
| WinoMT | `translation_data/mt_gender` | Yes |
| OntoNotes | `summarization_data/ontonotes` | Rebuild |
| BBQ | `conv_ai/bbq` | Yes |
| BOLD | `conv_ai/bold` | Yes |

## Run

Generate requests, submit them, then evaluate the saved responses:

```sh
llmbias build --dataset medical_data/medbullets --model gpt-4.1-2025-04-14 \
  --output-dir runs/medbullets/requests
llmbias run submit-openai -- \
  --input runs/medbullets/requests/gpt-4.1-2025-04-14_medbullets.jsonl \
  --output runs/medbullets/responses.jsonl
llmbias run evaluate-medical -- \
  --pred runs/medbullets/responses.jsonl \
  --gt data/processed/medical_data/medbullets.csv --model_tag medbullets
```

For configured workflows, supply a JSON file whose keys match the function's
arguments and whose values specify your input/output paths:

```sh
llmbias workflow evaluate-cams --config runs/cams.json
```

List workflows or inspect standalone command arguments:

```sh
llmbias list
llmbias workflow evaluate-cams
llmbias run submit-anthropic -- --help
llmbias run normalize-paper -- --help
llmbias run assemble-table -- --help
llmbias run evaluate-half -- --help
```

For summarization, set up the external scorer once:

```sh
python scripts/setup_summary_bias.py
python -m spacy download en_core_web_trf
python -m nltk.downloader punkt punkt_tab
```

## Code

```text
src/llmbias/
├── providers/                 # API clients
├── processing/
│   ├── preprocessing/         # Datasets and demographic variants
│   ├── requests/              # Prompts and request builders
│   └── parsing/               # Responses to predictions
├── data/                      # Dataset registry and paths
├── evaluation/                # Domain metrics, normalization and table assembly
└── cli.py                     # CLI entry point
```
