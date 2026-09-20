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

Datasets and supporting files are included in `data/processed/`.
Use these dataset keys with `llmbias build --dataset`:

| Dataset | Key |
| --- | --- |
| MedBullets | `medical_data/medbullets` |
| BiasMedQA | `medical_data/medical_bias` |
| CAMS | `mental_health_data/CAMS` |
| SAD | `mental_health_data/SAD` |
| Djinni | `admission_data/djinni` |
| Education | `education_data/education_ranking` |
| MovieLens | `recommendation_system/movielens` |
| ECtHR | `legal_data/ecthr` |
| WinoMT | `translation_data/mt_gender` |
| OntoNotes | `summarization_data/ontonotes` |
| BBQ | `conv_ai/bbq` |
| BOLD | `conv_ai/bold` |

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

For other tasks, edit paths and replace `${PLACEHOLDER}` values in an
[example configuration](examples/), then run the corresponding workflow:

```sh
llmbias workflow parse-cams-openai --config examples/cams-parsing.json
llmbias workflow evaluate-cams --config examples/cams-score.json
llmbias workflow evaluate-bold --config examples/bold-evaluation.json
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
