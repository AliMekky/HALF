# Active experiment scope

Scope follows the supplied `ARR_Oct_LLMBias (2).pdf`: evaluated experiments, including appendix results, rather than every dataset mentioned in related work. The active package contains these twelve datasets:

| Package key | Paper experiment | Results |
| --- | --- | --- |
| `medbullets` | MedBullets | Medical results |
| `medical_bias` | BiasMedQA | Medical cognitive-bias results |
| `ecthr` | ECtHR / FairLex | Legal results |
| `CAMS` | CAMS | Mental-health results |
| `SAD` | SAD | Mental-health results |
| `djinni` | Djinni recruitment | Recruitment results |
| `movielens` | MovieLens 20M recommendations | Appendix B.5, Table 9 |
| `education_ranking` | Education explanation selection, student/teacher roles | Appendix B.6, Tables 10–11 |
| `mt_gender` | WinoMT / MT-Gender | Table 12 |
| `bbq` | BBQ | Table 13 |
| `bold` | BOLD | Table 14 |
| `ontonotes` | OntoNotes / SummaryBias | Table 15 |

Table 1 lists eleven datasets, while the appendix additionally reports MovieLens. MovieLens therefore remains active. DiaSafety is cited in related work but has no evaluated experiment in the PDF.

Removed from active registration, dedicated modules/functions, and preparation recipes: Dreaddit, education admission (`education_ga`), DiaSafety, BiasMD (distinct from BiasMedQA), DiseaseBuster, IRS joint-label sampling, and admission-data sampling/field extraction. The inactive education-generation prompt entry was removed too. Shared preparation for retained datasets and historical variants of their evaluators remain available; removing a dataset does not justify changing a published task's methods.

No original files or archival results were deleted. Discarded experiments remain only in the ignored historical `legacy/` archive and prior Git history. Historical review documents describe the earlier inventory; this file and `llmbias coverage` define the current scope. [Archived entries](archived-experiments.json) record the removal.

[Recovered BOLD scoring](bold.md) now uses workspace `conv_ai/eval.py`, including its `unitary/toxic-bert` model and request-ID category aggregation. Exact historical library/model revisions remain unrecorded in the source; full toxicity inference has not been rerun. Summarization still requires its original upstream NLP environment.
