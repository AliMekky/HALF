# Finding the code

The execution flow is:

**data → preprocessing → requests → provider → parsing → evaluation**

| Responsibility | Location | Example |
| --- | --- | --- |
| Submit requests and retrieve raw responses | `providers/` | `openai.py`, `anthropic.py` |
| Prepare datasets and demographic variants | `processing/preprocessing/` | `datasets.py`, `demographics.py` |
| Build requests and prompts | `processing/requests/` | `medical.py`, `prompts.py` |
| Parse provider responses | `processing/parsing/` | `cams_openai.py`, `medical_answer.py` |
| Identify datasets and locate files | `data/` | `registry.py`, `paths.py` |
| Score and report results | `evaluation/<domain>/` | `recruitment/scoring.py`, `recruitment/reporting.py` |

These paths are relative to `src/llmbias/`. The root-level `data/processed/`
directory holds actual dataset files; `src/llmbias/data/` holds Python code only.

## Evaluation domains

| Domain | Main modules |
| --- | --- |
| Medical | `scoring.py` (MedBullets), `bias.py` (BiasMedQA), `accuracy.py`, `qualitative.py` |
| Mental health | `cams.py`, `cams_groups.py`, `sad.py`, `sad_workflow.py`, `intersectional.py`, `reporting.py` |
| Recruitment | `scoring.py`, `workflow.py`, `reporting.py` |
| Education | `scoring.py`, `workflow.py` |
| Recommendation | `scoring.py`, `metrics.py` |
| Legal | `scoring.py`, `tensors.py`, `reporting.py` |
| Translation | `scoring.py` |
| Summarization | `scoring.py` (upstream scorer adapter) |
| Conversational | `bbq.py`, `bold.py`, `bold_metrics.py` |

`workflow.py` modules connect explicit file inputs and output directories to
scoring functions. `reporting.py` modules produce domain reports. Functions
remain separate where combining them would obscure distinct published behavior.

## Entry points

`evaluation/normalization.py` normalizes domain metrics with explicit calibration
policies. `evaluation/table_assembly.py` reduces legal, recruitment and education
outputs and formats verified demographic deltas. See [usage and evidence](paper-reconstruction.md).

`evaluation/aggregation.py` reconstructs the cross-domain HALF formula from
already normalized scores. It is separate from the preserved domain scorers;
see [metric provenance](metric-provenance.md) for its verification limits.

`cli.py` exposes `build`, `run`, `workflow`, `list`, and `coverage`.
`_workflows.py` is internal dispatch/configuration code; it contains no scientific
metric implementation. Existing CLI command names are unchanged.

For Python usage, import from the new locations:

```python
from llmbias.data.registry import functions
from llmbias.processing.requests.medical import create_med_bullets_template
from llmbias.evaluation.recommendation.metrics import jaccard, prag
```

The previous flat imports are replaced by these locations. There are no duplicate
compatibility modules. Older implementations remain in the original repository.

Parser filenames use `task_provider.py` (for example `mcq_anthropic.py` and
`cams_openai.py`). Shared response conversion lives in `processing/parsing/response_files.py`.
Summary/input merging lives in `processing/parsing/summarization_merge.py`.
