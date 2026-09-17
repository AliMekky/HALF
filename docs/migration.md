# Structural migration

The source is the existing `repo/LLMBias` working tree, including modified and untracked files. This preserves the research state available locally rather than assuming that its Git HEAD contains the accepted experiment.

## Allowed changes made

- Split `experiments/utils.py` into domain task modules and demographic interventions; retain its registry and a compatibility import facade.
- Move provider submission, parsing, scoring, and dataset preparation scripts into named modules.
- Update internal imports. Relocate absolute workspace paths in standalone modules through `llmbias.paths.workspace_path`.
- Extract named notebook functions into separate analysis modules without executing cells or rewriting function logic.
- Replace embedded credential literals with environment-variable lookups; redact credential strings from the local archive.
- Add packaging, documentation, provenance, an offline request-generation wrapper, and preservation checks.

The new `build` wrapper supports an explicit output directory and refuses to overwrite existing files. The historical builder is also available through `run create-batch` with its original arguments and relative directory layout.

## Intentionally unchanged

Prompt strings, demographic order, first-500 selection, IDs, temperature rules, neutralization behavior, answer parsing, grouping, refusal handling, metric formulas, and existing candidate correctness issues were not revised. No published output was recalculated or replaced. All noncredential artifacts in the local snapshot retain their original content; JSON serialization of the archived notebook is normalized.

## Historical workflows

Some notebook functions use module globals; explicit adapters expose their input and output paths. Preparation script execution is guarded by main entry points. The full notebook and original scripts are in `legacy/`. Extracted analysis function provenance includes zero-based notebook cell numbers. The later recommendation functions come from cell 382; the earlier cell 379 is also available as a separately named historical workflow.

The legacy archive keeps nested upstream code without their `.git` histories. Their license/README files remain. External workspace inputs outside LLMBias are not duplicated; scripts needing them require `LLMBIAS_WORKSPACE`. This includes translation source files at the enclosing workspace root. Moving to another machine requires restoring those inputs as well as the artifact bundle.

## Validation scope

Checks compare extracted function syntax trees and prompt constants with the preserved source, and compare generated requests on actual dataset samples for every registered task. Small synthetic tests compare scoring/parsing functions with their original notebook or script implementations. These are equivalence checks, not new assertions about scientific validity. No live API jobs or complete historical notebook execution are used.

The artifact inventory records whether bytes are identical; differences are limited to text containing credentials and normalized notebook JSON. Verification also checks the original source files against their recorded hashes after migration.

## Paper-scoped follow-up

The active package now follows [paper scope](paper-scope.md), removing unreported experiments while keeping the archive intact. The BOLD evaluator is new reference-based code documented in [BOLD scoring](bold.md), rather than an unchanged notebook extraction. Existing request builders and evaluator formulas for retained tasks remain preserved.
