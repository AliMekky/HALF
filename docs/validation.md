# Initial migration validation

Historical record of the first migration. See [BOLD and paper-scope validation](bold-validation.json) for the subsequent implementation pass.

The offline preservation suite passed all nine tests:

- All 14 registered dataset builders produced the same Python request objects on three-row samples from the saved datasets.
- The new batch wrapper produced byte-identical JSONL to the original CLI for sampled Medbullets inputs with `gpt-4o` and `o4-mini-2025-04-16` settings.
- Extracted task and notebook function syntax trees match their originals, normalizing only filesystem relocation expressions.
- Moved standalone script function syntax trees match the credential-sanitized legacy copy, normalizing only filesystem relocation expressions.
- All uppercase prompt/configuration constants match.
- Representative OpenAI, Anthropic, and DeepSeek parsers match.
- Recruitment grouping, later recommendation metrics, CAMS equalized-odds computation, and SAD F1 match on synthetic inputs.

All 50 package Python files parsed successfully. CLI discovery and the relocated accuracy evaluator's help command worked. The 827 inventoried original files were checked against their recorded SHA-256 hashes, with no source changes. Artifact-copy details are recorded in `artifact-verification.json`.

These checks establish structural preservation for the tested paths. They do not establish scientific correctness of the inherited code, identify which notebook version produced every published number, validate live provider compatibility, or reproduce the full paper. No API requests, paid inference, historical notebook execution, or published-result replacement occurred.

Run again after restoring the local artifact bundle:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```
