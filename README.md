# HALF

## Install

Use Python 3.10 or newer. From the repository root:

```sh
conda activate bias
python -m pip install -c requirements/constraints.txt -e '.[providers,analysis,datasets,bold-model,summarization]'
```

To create the environment first: `conda create -n bias python=3.10`.
Export the API credentials for your provider using the names in [.env.example](.env.example).

## Data

Place datasets in `data/processed/` using the paths in [data/README.md](data/README.md),
or import them from the original repository:

```sh
python scripts/import_datasets.py --source /path/to/LLMBias
```

Use `--data-root /path/to/data` with `llmbias build` to select another directory.

## Run

Build requests, submit them, then evaluate the saved responses. For MedBullets:

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

For other datasets, choose a configuration from `examples/` or
`experiments/configs/`, replace its input/output paths, and run its workflow:

```sh
llmbias workflow parse-cams-openai --config examples/cams-parsing.json
llmbias workflow evaluate-translation --config examples/translation-evaluation.json
```

Find commands and arguments with:

```sh
llmbias list
llmbias run submit-anthropic -- --help
llmbias workflow evaluate-cams
```

[Workflow arguments](docs/workflows.md) ·
[Normalization and table assembly](docs/paper-reconstruction.md)

For summarization, install the scorer and language resources once:

```sh
python scripts/setup_summary_bias.py
python -m spacy download en_core_web_trf
python -m nltk.downloader punkt punkt_tab
```

## Layout

```text
src/llmbias/
├── providers/                 # OpenAI, Anthropic, DeepSeek, DeepInfra
├── processing/
│   ├── preprocessing/         # Dataset preparation and demographic variants
│   ├── requests/              # Prompts and request builders
│   └── parsing/               # Response conversion and prediction extraction
├── data/                      # Dataset registry and paths
├── evaluation/                # Domain scorers, normalization and table assembly
└── cli.py                     # Command-line interface

data/processed/                # Input datasets
examples/                      # Workflow configurations
experiments/configs/           # Dataset evaluation configurations
scripts/                       # Setup, import and validation utilities
runs/                          # Generated requests, responses and results
```

## Tests

```sh
python -m unittest discover -s tests -q
python scripts/smoke_test.py
```

## Citation

[Paper](https://arxiv.org/abs/2510.12217) · [BibTeX](CITATION.bib) · [Citation metadata](CITATION.cff)
