# Data

HALF evaluates models on 12 datasets. Ten are included in `data/processed/`.
MovieLens and OntoNotes are **not included** because their licences do not
allow redistribution. You need to obtain them from their original sources and rebuild the
processed files with `scripts/restore_data.py`, which checks every rebuilt file
against the SHA-256 recorded in `datasets-manifest.json`.

Every dataset keeps its original licence. The MIT licence of this repository
covers only our code. Please cite the original dataset papers (see the HALF
paper, Appendix C) when you use any of them.

## Included datasets

| Dataset | Key | Original source | Licence / terms |
| --- | --- | --- | --- |
| CAMS | `mental_health_data/CAMS` | [drmuskangarg/CAMS](https://github.com/drmuskangarg/CAMS) (Garg et al., 2022) | Free for research use, with citation (CAMS terms of use) |
| SAD | `mental_health_data/SAD` | [PervasiveWellbeingTech/Stress-Annotated-Dataset-SAD](https://github.com/PervasiveWellbeingTech/Stress-Annotated-Dataset-SAD) (Mauriello et al., 2021) | MIT |
| Djinni | `admission_data/djinni` | [lang-uk recruitment datasets](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english) (Drushchak and Romanyshyn, 2024) | MIT (data); code in [Stereotypes-in-LLMs/recruitment-dataset](https://github.com/Stereotypes-in-LLMs/recruitment-dataset) is Apache-2.0 |
| Education | `education_data/education_ranking` | Weissburg et al. (2025), [Findings of NAACL](https://aclanthology.org/2025.findings-naacl.314/) | MIT, as stated by the authors |
| ECtHR (FairLex) | `legal_data/ecthr` | [coastalcph/fairlex](https://huggingface.co/datasets/coastalcph/fairlex) (Chalkidis et al., 2022) | **CC BY-NC-SA 4.0**. Our processed file, including the imputed applicant gender, is shared under the same licence: non-commercial use only |
| WinoMT | `translation_data/mt_gender` | [gabrielStanovsky/mt_gender](https://github.com/gabrielStanovsky/mt_gender) (Stanovsky et al., 2019) | MIT |
| BBQ | `conv_ai/bbq` | [nyu-mll/BBQ](https://github.com/nyu-mll/BBQ) (Parrish et al., 2022) | CC BY 4.0 |
| BOLD | `conv_ai/bold` | [amazon-science/bold](https://github.com/amazon-science/bold) (Dhamala et al., 2021) | CC BY-SA 4.0. Our 1,000-prompt subset is shared under the same licence |
| MedBullets | `medical_data/medbullets` | [HanjieChen/ChallengeClinicalQA](https://github.com/HanjieChen/ChallengeClinicalQA) (Chen et al., 2025) | No licence stated by the source; shared here with our demographic annotations for research reproducibility |
| BiasMedQA | `medical_data/medical_bias` | [carlwharris/cog-bias-med-LLMs](https://github.com/carlwharris/cog-bias-med-LLMs) (Schmidgall et al., 2024), built on [MedQA](https://github.com/jind11/MedQA) | No licence stated by the BiasMedQA source (MedQA is MIT); shared here for research reproducibility |
| MedQA answers for BiasMedQA | `medical_data/medical_bias_gt.jsonl` | [jind11/MedQA](https://github.com/jind11/MedQA) (US test split) | MIT |

## Datasets you need to rebuild

Install the package first (see the main README), then run the commands from
the repository root.

### MovieLens (`recommendation_system/movielens`)

- **Why it is not included:** the
  [MovieLens licence](https://files.grouplens.org/datasets/movielens/ml-20m-README.html)
  says “The user may not redistribute the data without separate permission.”
- **Rebuild:** download
  [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) yourself,
  accepting its terms, then run the command below. The user sampling is seeded,
  so the output matches the paper.

  ```sh
  python scripts/restore_data.py movielens --ml20m-dir path/to/ml-20m
  ```

### OntoNotes / SummaryBias (`summarization_data/ontonotes`)

- **Why it is not included:** OntoNotes 5.0
  ([LDC2013T19](https://catalog.ldc.upenn.edu/LDC2013T19)) is licensed by the
  Linguistic Data Consortium, and the SummaryBias instances contain its
  newswire text.
- **What we ship:** `data/annotations/ontonotes_ids.csv`, which lists the
  article and pair IDs we used and contains no text.
- **Rebuild:**
  1. Obtain OntoNotes through LDC.
  2. Set up SummaryBias with `python scripts/setup_summary_bias.py`.
  3. Generate the instances from the OntoNotes `nw` directory by following the
     [SummaryBias instructions](https://github.com/julmaxi/summary_bias).
  4. Convert them:

     ```sh
     python scripts/restore_data.py ontonotes --summarybias-jsonl path/to/onto-nw_gender_balanced_1.jsonl
     ```

The script warns you if the rebuilt article and pair IDs differ from ours.
