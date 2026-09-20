# BOLD evaluation

`evaluate-bold` preserves the original project `conv_ai/eval.py` scoring logic.
It consumes raw provider response JSONL. Edit the example input path to the
responses from your run:

```sh
pip install -e '.[bold-model]'
llmbias workflow evaluate-bold --config examples/bold-evaluation-original.json
```

Examples also cover Anthropic, DeepSeek, and o4 responses. Pass responses, not
request batches; the wrapper rejects request-only inputs before model loading.
Outputs are `domain_metrics.csv`, `gender_polarity.csv`, `overall.json`, and
`provenance.json`, written into a fresh output directory.

## Preserved behavior

- **Sentiment:** VADER `compound` on the raw response, with no prompt prefix or anonymization.
- **Toxicity:** `transformers.pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)`. Select the score whose label lowercases to `toxic`; preserve the original fallback of 0.0 if that label is absent. No substitute model, six-label union, thresholding, or extra truncation is introduced.
- **Grouping:** use the original `bold-([^-]+)-` request-ID regex. Despite the column name `domain`, this extracts dataset **categories**, not the five broad BOLD topical domains. The saved LLaMA-8B table contains 41 groups. Retain `unknown` when extraction fails.
- **Gender polarity:** preserve the exact male/female term sets and token-count difference, then average within male/female/unknown actor-ID groups. This is the script's implementation, not a replacement gender metric from another source.
- **Parsing:** default `response_format="openai"`, `id_format="bold"` preserves the active code. `response_format="anthropic"` or `"deepseek"` and `id_format="short"` select the commented extraction alternatives found in the source. Formats are never silently inferred. Existing empty-text handling and parsing failure behavior remain intact.
- **Model settings:** no `device`, model revision, or truncation argument is supplied by default, matching the actual source call. The source's “force CPU” comment has no corresponding argument. Optional explicit `device=-1` or `model_revision` is recorded in provenance rather than described as an original setting.

`evaluation/conversational/bold_metrics.py` holds the extracted calculation functions and loop; `evaluation/conversational/bold.py` provides dependency initialization, paths, output protection, and run metadata. The three metric function bodies are AST-equivalent to their originals. Classifier objects can be injected into the calculation function for deterministic preservation tests.

## Overall reporting

The original script exports group tables. The wrapper reports their unweighted
mean, matching the supplied LLaMA-8B saved aggregate at the paper's precision.
This aggregation is inferred from the implementation and saved outputs; it differs
from the PDF's wording about five topical domains. The implementation is preserved.
Full toxicity inference across all models has not been rerun. Original model
revisions were not recorded; new runs record the resolved revision when available.
