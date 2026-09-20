# Installation and dependencies

Use Python 3.10 or newer. The existing project environment is Conda `bias`
(Python 3.10.16). Its observed package versions are in
[bias-environment.json](bias-environment.json); this is an inventory, not a
portable lock file or proof of the original experiment-time versions.

For a new environment:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -c requirements/constraints.txt -e '.[providers,analysis,datasets,bold-model,summarization]'
llmbias coverage
```

Install only needed extras: `providers` for API clients, `analysis` for plotting
and Torch-based legal evaluation, `datasets` for Parquet and multilabel sampling,
`bold-model` for BOLD, and `summarization` for SummaryBias. The inspected `bias`
environment lacks VADER and NLTK; these extras declare them. Export the credentials
in `.env.example` before using provider submission commands.

## Data and outputs

The [dataset guide](../data/README.md) specifies all paper input files and how to
import them. `llmbias build` defaults to `data/processed/`. Use `--data-root` or
`LLMBIAS_DATA_ROOT` to select another location. Workflow configurations accept
explicit file paths. Save generated requests, provider responses, and results
under `runs/`; `LLMBIAS_RUNS_ROOT` changes defaults for file-based helper scripts.
No command requires the old repository once the required inputs are present.

## Summarization

The required scorer comes from Julian Steen and Katja Markert's
[SummaryBias repository](https://github.com/julmaxi/summary_bias), pinned to
`bac43fbea0a30ad5f97328ce4bfc7ff2f23cf7f8`. A local copy of its tracked files is
available under `third_party/summary_bias/`. It excludes generated summaries and
Git history. For a fresh clone, fetch it once:

```sh
python scripts/setup_summary_bias.py
python -m spacy download en_core_web_trf
python -m nltk.downloader punkt punkt_tab
```

Skip the setup script if the scorer directory already exists. It refuses to
replace an existing directory. Its patch restores the original local NLTK
resource-loading change; metric calculations are unchanged.
`evaluate-summarization` defaults to this scorer directory. `upstream_root` and
`python` can select another checkout and NLP environment. OntoNotes inputs must
be obtained under their own access terms; they are not included in a public clone.

## BBQ and BOLD

BBQ needs `data/processed/conv_ai/bbq_additional_metadata.csv`; the importer copies
the original scorer's metadata variant. The adapted scorer and its test fixture
originate from [NYU BBQ](https://github.com/nyu-mll/BBQ),
`analysis_scripts/BBQ_bias_score.py`. File boundaries and input/output paths were
adapted. Its [CC BY 4.0 license](../third_party/notices/BBQ-LICENSE) is retained.

BOLD downloads `unitary/toxic-bert` when invoked; see [BOLD scoring](bold.md).
Full NLP/model inference is not part of the offline test suite.

## Release verification

The core release was installed in a clean Python 3.10 environment and all 48
offline tests passed, including VADER. `scripts/smoke_test.py` checks request
building, response parsing and medical scoring using synthetic temporary data.
Run it after installing the package:

```sh
python scripts/smoke_test.py
python scripts/paper_manifest.py validate
```

`requirements/core.txt` pins the core direct dependencies.
`requirements/core-lock.txt` records the complete tested core environment
(excluding the project and build tools). Optional direct dependencies are
constrained separately in `requirements/constraints.txt`; heavyweight model
inference and all optional extras are not covered by the core smoke test.

The GitHub Actions workflow runs offline tests, manifest validation and the smoke
test on Python 3.10. Data, model downloads and API credentials are not needed.
