# LLMBias Research

A separate, modular working repository for the existing LLMBias research code. This is a structural migration of the current working tree, including untracked research files. It does not revise published methods or claim to reproduce every paper table.

For retained historical workflows, scientific function bodies, prompts, dataset selection, demographic combinations, parsing rules, model settings, and metric formulas are preserved. Active experiment scope now follows the supplied paper; the new BOLD evaluator is a documented reference-based reconstruction. Changes are module boundaries, imports, filesystem configuration, documentation, and environment-based credential loading. Suspected methodological issues are deliberately left unchanged.

See [paper scope](docs/paper-scope.md) for the twelve retained experiments and [BOLD scoring](docs/bold.md) for its original sources, commands, and reproduction limits.

## Layout

```text
src/llmbias/
  tasks/           Domain-specific request builders
  interventions/   Existing demographic transformations
  prompts.py       Original prompt text and demographic configurations
  registry.py      Original dataset-to-builder mapping
  providers/       Existing submission implementations
  parsing/         Existing provider-specific extraction scripts
  preparation/     Existing data preparation scripts
  evaluation/      Standalone scoring implementations
  analysis/        Versioned notebook analysis functions
  reporting/       Historical plots and qualitative enrichment
  workflows.py     Explicit notebook workflow registry and task coverage
  pipeline/        Historical batch-generation entry point
  cli.py           Unified entry points and offline request generation
legacy/            Independent local copy of data, outputs, upstream code, notebook
data/              Artifact checksums and source inventory
docs/              Migration map, provenance, and limitations
notebooks/         Guide to the preserved notebook
tests/             Offline behavior-preservation checks
```

`legacy/` is intentionally excluded from Git because it contains about 3.4 GB of datasets, raw outputs, and third-party research material. It is copied locally, not symlinked to the original repository. The manifest records each source/copy checksum. A Git clone alone will need this artifact bundle restored into `legacy/`, or `LLMBIAS_LEGACY_ROOT` set to its location. No GitHub repository has been created or published.

## Setup

Use Python 3.10 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[providers,analysis,datasets]'
llmbias list
```

For an existing environment, no installation is needed to try the local package:

```sh
PYTHONPATH=src python -m llmbias.cli list
PYTHONPATH=src python -m unittest discover -s tests -v
```

The dependency declarations were inferred from imports; the historical repository did not supply a complete environment lock. `docs/validation-environment.json` records the environment used for migration checks, not the environment of the original paper. Upstream tools under `legacy/experiments/*/evaluation/` retain their own requirements and licensing notices.

## Generate the same requests, offline

```sh
PYTHONPATH=src python -m llmbias.cli build \
  --dataset medical_data/medbullets \
  --model gpt-4o \
  --output-dir runs/medbullets-example
```

The new wrapper uses the original builders and batch settings. Existing output files are protected from accidental overwriting. It makes no API calls. Other dataset names retain their historical directory prefixes, such as `mental_health_data/CAMS`, `recommendation_system/movielens`, and `education_data/education_ranking`.

## Use individual modules

```python
from llmbias.tasks.medical import create_med_bullets_template
from llmbias.evaluation.accuracy import compute_accuracy
from llmbias.analysis.recommendation import jaccard, prag
```

The provider, parser, and evaluator scripts retain their original arguments:

```sh
PYTHONPATH=src python -m llmbias.cli run evaluate-accuracy -- --help
PYTHONPATH=src python -m llmbias.cli run parse-openai -- --help
```

Run provider commands only when you intend to submit paid inference jobs. Export credentials named in `.env.example`; no credentials were retained in the new source. Preparation script execution is guarded by main entry points. Workflow discovery uses a lazy registry.

## Notebook workflows

```sh
PYTHONPATH=src python -m llmbias.cli coverage
PYTHONPATH=src python -m llmbias.cli workflow evaluate-translation --config examples/translation-evaluation.json
```

See [workflow instructions](docs/workflows.md) for configuration, historical versions, and remaining source gaps. Standalone notebook preservation tests include their own sanitized source fixtures and do not require the legacy bundle.

## Preservation and provenance

- [Migration details](docs/migration.md): what moved and what remains historical.
- [Module map](docs/module-map.json): original file/function/notebook cell to new module.
- [Analysis usage](docs/analysis.md): historical module-global dependencies.
- [Artifact manifest](data/legacy-manifest.json): original and copied checksums.
- [Source provenance](docs/source.json): original working tree and Git HEAD.

The original repository has not been edited. The preserved notebook includes saved outputs, but code cells loading credentials now use environment variables and credential strings in outputs are redacted. The notebook is an archive, not a clean Run All pipeline. Neither it nor paid API workflows were executed during migration.

Future methodological improvements should be separately versioned. This migration intentionally does not repair legacy parsing, pairing, missing-data, or metric behavior. A clean package layout is not evidence that a published table has been independently reproduced.

No new license or paper citation metadata is invented here. Confirm the authors' intended project license and supply the accepted paper's citation before public release. Upstream notices remain with the preserved dependencies.
