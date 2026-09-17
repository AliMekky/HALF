# Using extracted notebook functions

Extraction keeps the function bodies and the historical distinctions between analysis versions. All cell numbers below are zero-based. The source map records provenance for each function.

| Module | Cell | Typical entry point |
| --- | --- | --- |
| `analysis.recruitment` | 306 | `add_reference_columns`, `summarise` |
| `analysis.cams` | 338 | `main(gold_path, neutral_path, sensitive_path, out_path)` |
| `analysis.cams_groups` | 343 | `compute_all_group_metrics(gold_csv, model_configs)` |
| `analysis.sad` | 358 | `evaluate_model` |
| `analysis.education` | 367 | `compute_bias_by_role` |
| `analysis.recommendation` | 382 | `extract_items`, `jaccard`, `prag` |
| `analysis.translation_judge` | 451 | `create_mt_gender_batch` |
| `analysis.qualitative` | 490 | `flip_stats`, `main` |

For recruitment, the original `MODEL_NAMES`, `DATA_DIR`, and `OUT_CSV` remain module-level configuration. `DATA_DIR` points to the copied legacy outputs by default. For another study:

```python
from pathlib import Path
from llmbias.analysis import recruitment

recruitment.DATA_DIR = Path("my-study/predictions")
recruitment.OUT_CSV = Path("my-study/recruitment_metrics.csv")
recruitment.MODEL_NAMES = ["my-model"]
# recruitment.main()  # Executes the historical analysis with those paths.
```

Qualitative analysis likewise retains `DATA_DIR`, `REASONING_NAME`, `REASONING_FILE`, `STANDARD_MODELS`, and prediction-column configuration. Their historical values are preserved, with the default data directory relocated. SAD retains its original `LABELS` global, even where the legacy function also accepts a label argument. Education retains its module-level seeded RNG, including its stateful behavior across calls. Resetting these behaviors would be a methodological change and is outside this migration.

The older recommendation implementation remains in the full notebook; the reusable functions are from the later cell 382. This selection does not establish which implementation produced a specific paper table. Scripts needing exact historical notebook state should use the archived notebook and its saved outputs as reference, not assume that extracted functions reproduce all cell execution history.
