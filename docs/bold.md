# Recovered BOLD evaluation

`evaluate-bold` now runs the implementation recovered from **workspace `conv_ai/eval.py`**, outside the previously inspected `repo/LLMBias` directory. Its source is preserved in `tests/fixtures/bold_eval_original.py.txt`; a hash-verified local copy of the entire newly supplied folder is in ignored `legacy_additions/conv_ai/`. The original folder is untouched.

## Run

```sh
pip install -e '.[bold-model]'
llmbias workflow evaluate-bold --config examples/bold-evaluation-original.json
# Other saved response formats, explicitly selected from alternatives in eval.py:
llmbias workflow evaluate-bold --config examples/bold-evaluation-claude.json
llmbias workflow evaluate-bold --config examples/bold-evaluation-deepseek.json
```

The original workflow consumes **raw response JSONL**, not the normalized records used by the earlier reconstruction. Restore `legacy_additions/` when using a fresh clone, or edit the input path. Optional model downloads occur only when evaluation is invoked. Imports and workflow discovery do not load a model or submit inference requests to generation providers.

**o4-mini file selection:** in this newly supplied folder, `o4-mini-2025-04-16_bold.jsonl` contains generation requests; the corresponding responses are in `o4-mini-2025-04-16_bold_copy.jsonl`. Use `examples/bold-evaluation-o4.json`. The wrapper rejects request batches before loading the model.

All outputs go to a fresh `output_dir`: original `domain_metrics.csv` and `gender_polarity.csv`, plus the new `overall.json` and `provenance.json` reporting files.

## Preserved behavior

- **Sentiment:** VADER `compound` on the raw response, with no prompt prefix or anonymization.
- **Toxicity:** `transformers.pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)`. Select the score whose label lowercases to `toxic`; preserve the original fallback of 0.0 if that label is absent. No substitute model, six-label union, thresholding, or extra truncation is introduced.
- **Grouping:** use the original `bold-([^-]+)-` request-ID regex. Despite the column name `domain`, this extracts dataset **categories**, not the five broad BOLD topical domains. The saved LLaMA-8B table contains 41 groups. Retain `unknown` when extraction fails.
- **Gender polarity:** preserve the exact male/female term sets and token-count difference, then average within male/female/unknown actor-ID groups. This is the script's implementation, not a replacement gender metric from another source.
- **Parsing:** default `response_format="openai"`, `id_format="bold"` preserves the active code. `response_format="anthropic"` or `"deepseek"` and `id_format="short"` select the commented extraction alternatives found in the source. Formats are never silently inferred. Existing empty-text handling and parsing failure behavior remain intact.
- **Model settings:** no `device`, model revision, or truncation argument is supplied by default, matching the actual source call. The source's “force CPU” comment has no corresponding argument. Optional explicit `device=-1` or `model_revision` is recorded in provenance rather than described as an original setting.

`evaluation/bold_original.py` holds the extracted calculation functions and loop; `evaluation/bold.py` provides dependency initialization, paths, output protection, and run metadata. The three metric function bodies are AST-equivalent to their originals. Classifier objects can be injected into the calculation function for deterministic preservation tests.

## Relationship to Table 14

The original script exports group tables and does not implement a final overall mean. The new reporting wrapper takes an **unweighted arithmetic mean of its group means**. Evidence for this reporting choice:

- Mean of the supplied LLaMA-8B `domain_metrics.csv`: sentiment **0.12364867799499465**, toxicity **0.0011971413677312318**. These round to Table 14's **0.124** and **1.20 × 10⁻³**.
- VADER was rerun with the recovered parser/grouping on all eight supplied primary response files, including DeepSeek. All eight sentiment values match Table 14 at three decimal places. Results are recorded in [validation](bold-recovered-validation.json).
- The rerun LLaMA-8B per-group sentiment and gender-polarity tables agree with the supplied CSVs.

The inferred overall aggregation is labeled separately from the verbatim source logic. It differs from the PDF's stated macro-average over five topical domains; this integration preserves the recovered implementation rather than changing it to match the prose. The earlier mismatch with Table 14 came from using that five-domain reconstruction.

**Full toxicity inference across all eight models has not been rerun.** Deterministic tests compare the original and extracted scripts' CSV bytes using the same controlled classifier, including the exact model initialization arguments and toxic-label selection. Saved toxicity aggregates provide evidence for the LLaMA-8B row only. The source does not pin package versions or a model revision; each new real run records the resolved revision when available. Matching the saved aggregate is not proof of a new checkpoint inference reproduction.

## Earlier reconstruction

The provisional BOLD implementation remains explicitly named `evaluate-bold-reference`, `evaluate-bold-reference-sentiment`, and `convert-bold-reference`, with its code in `evaluation/bold_reference.py`. It is not the default or the historical evaluator. Its explicit metadata overrides, anonymization, and BERT-Large checkpoint adapter do not apply to `evaluate-bold`.

See [the superseded reference notes](bold-reference.md) for that separate implementation. The external original BOLD paper remains relevant background, but the recovered project script establishes the actual evaluation choices used here.
