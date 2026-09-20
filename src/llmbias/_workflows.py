"""Explicit entry points for historical notebook workflows.

Imports are lazy. Listing coverage or inspecting a command never loads data,
downloads models, or executes a notebook cell.
"""
from dataclasses import dataclass
import importlib
import inspect
import json
from pathlib import Path


@dataclass(frozen=True)
class Workflow:
    target: str
    source: str
    frames: tuple = ()
    frame_lists: tuple = ()
    note: str = ""


WORKFLOWS = {}


def register(name, target, source, frames=(), frame_lists=(), note=""):
    WORKFLOWS[name] = Workflow(target, source, frames, frame_lists, note)


for task, cells in [("cams", [328, 330, 332]), ("sad", [349, 351, 356])]:
    for provider, cell in zip(["openai", "anthropic", "deepseek"], cells):
        register(f"parse-{task}-{provider}", f"processing.parsing.{task}_{provider}:process_file", f"cell {cell}")
register("parse-recruitment-openai", "processing.parsing.recruitment_openai:process_file", "cell 302")
register("parse-recruitment-anthropic", "processing.parsing.recruitment_anthropic:process_file", "cell 300")
register("parse-education", "processing.parsing.education:process_jsonl", "cell 364", note="Provider detection retains historical filename conventions.")
register("parse-medical-answer", "processing.parsing.medical_answer:process_file", "cell 297")
register("prepare-medical-bias-neutral", "processing.requests.medical_bias_neutral:create_batch", "cell 265")
register("evaluate-medical-bias-neutral", "evaluation.medical.bias:evaluate_neutral", "cells 271,273")
register("evaluate-medical-bias-blocks", "evaluation.medical.bias:evaluate_blocks", "cells 275–280", note="Retains seven blocks of 1,273 responses.")
register("evaluate-medical-bias-matching", "evaluation.medical.bias:evaluate_prompt_matching", "cells 437,442,443")
register("evaluate-medical-bias-cleaned", "evaluation.medical.bias:evaluate_cleaned", "cell 287", frames=("all_prediction",))
register("evaluate-cams", "evaluation.mental_health.cams:main", "cell 338")
register("evaluate-cams-groups", "evaluation.mental_health.cams_groups:compute_all_group_metrics", "cell 343")
register("report-cams-intersectional", "evaluation.mental_health.reporting:report_cams_intersectional", "cell 341", note="Requires analysis extras; all artifacts go into output_dir.")
register("evaluate-sad", "evaluation.mental_health.sad_workflow:evaluate_sad", "cell 358")
register("evaluate-recruitment", "evaluation.recruitment.workflow:evaluate_recruitment", "cell 306")
register("evaluate-education", "evaluation.education.workflow:evaluate_education", "cell 367")
register("evaluate-recommendation", "evaluation.recommendation.scoring:evaluate", "cell 382")
register("prepare-translation-judge", "processing.requests.translation_judge:create_batch", "cells 451,452")
register("evaluate-translation", "evaluation.translation.scoring:evaluate", "cell 456")
register("convert-summarization", "processing.parsing.summarization:process_file", "cell 412")
register("evaluate-summarization", "evaluation.summarization.scoring:evaluate", "cell 414 and preserved summary_bias checkout", note="Requires external NLP environment/resources; explicitly runs upstream tools.")
register("evaluate-bold", "evaluation.conversational.bold:evaluate", "workspace conv_ai/eval.py", note="Original raw-response VADER/unitary-toxic-bert scoring; see docs/bold.md.")
register("convert-bbq", "processing.parsing.bbq:process_file", "cell 420")
register("evaluate-bbq", "evaluation.conversational.bbq:evaluate", "preserved BBQ/analysis_scripts/BBQ_bias_score.py")
register("prepare-legal-gender", "processing.requests.legal_gender:create_batch", "cell 386")
register("prepare-legal-placeholders", "processing.requests.legal_placeholders:create_batch", "cell 395", note="Creates placeholder-detection requests, not an ethnicity intervention.")
register("reformat-claude", "processing.parsing.response_files:reformat_claude", "cell 448")
register("concat-responses", "processing.parsing.response_files:concat_jsonl_files", "cell 472")

for name, fn, cells, frames, frame_lists in [
    ("education", "prepare_education", "100–104", (), ("frames",)),
    ("bbq", "prepare_bbq", "188", ("df",), ()),
    ("recruitment", "match_recruitment", "153", ("df_cv", "df_jobs"), ()),
    ("movielens", "prepare_movielens", "220,222,226", ("ratings", "movies"), ()),
    ("medical-bias-sample", "sample_medical_bias", "48", ("merged_df",), ()),
    ("mental-multilabel", "sample_mental_multilabel", "71", ("df",), ()),
    ("translation", "prepare_translation", "111,114,115,117", (), ()),
    ("neutralization-results", "attach_neutralized_text", "322,323", ("df",), ()),
    ("legal-gender-results", "update_legal_gender", "387", (), ()),
    ("ontonotes", "prepare_ontonotes", "403", (), ()),
    ("medbullets", "prepare_medbullets", "31,32,37", ("df",), ()),
    ("medbullets-sample", "sample_medbullets_by_gender", "40", ("df",), ()),
    ("medical-prompts", "prepare_medical_prompts", "213,216", (), ()),
]:
    register("prepare-" + name, "processing.preprocessing.datasets:" + fn, "cells " + cells, frames, frame_lists)


register("report-recruitment", "evaluation.recruitment.reporting:report", "cell 308")
register("report-legal", "evaluation.legal.reporting:summarize_metadata", "cell 418")
register("enrich-medbullets", "evaluation.medical.qualitative:enrich_medbullets", "cell 429", frames=("outputs", "gt"))
register("enrich-deepseek", "evaluation.medical.qualitative:enrich_deepseek", "cell 433", frames=("outputs",))
register("enrich-flip-cases", "evaluation.medical.qualitative:enrich_flip_cases", "cells 495–497", frames=("detailed",))

# A dataset builder is not evidence of an evaluator. These statuses describe
# implementations located in the available source, not paper-table provenance.
TASK_COVERAGE = {
    "medbullets": {"generation": True, "evaluation": "standalone medical scorer", "parsing": "existing MCQ parsers; medical cleanup"},
    "medical_bias": {"generation": True, "evaluation": "neutral, blocks, matching, cleaned; explicitly versioned", "parsing": "existing MCQ parsers; medical cleanup"},
    "CAMS": {"generation": True, "evaluation": "evaluate-cams / evaluate-cams-groups", "parsing": "three provider parsers"},
    "SAD": {"generation": True, "evaluation": "evaluate-sad", "parsing": "three provider parsers; onehot and indices"},
    "djinni": {"generation": True, "evaluation": "evaluate-recruitment", "parsing": "OpenAI and Anthropic; no dedicated DeepSeek decision parser located"},
    "education_ranking": {"generation": True, "evaluation": "evaluate-education", "parsing": "parse-education"},
    "movielens": {"generation": True, "evaluation": "evaluate-recommendation", "parsing": "inside evaluator, provider/filename dependent"},
    "ecthr": {"generation": True, "evaluation": "standalone legal tensors and fairness evaluator", "parsing": "inside legal tensor preparation"},
    "mt_gender": {"generation": True, "evaluation": "evaluate-translation", "parsing": "judge outputs; prepare-translation-judge uses historical judge settings"},
    "ontonotes": {"generation": True, "evaluation": "evaluate-summarization (external dependencies)", "parsing": "convert-summarization"},
    "bbq": {"generation": True, "evaluation": "evaluate-bbq", "parsing": "convert-bbq"},
    "bold": {"generation": True, "evaluation": "evaluate-bold (recovered conv_ai/eval.py; unitary/toxic-bert)", "parsing": "original provider-specific raw-response extraction", "paper": "Table 14; Appendix D.7"},
}


def resolve(name):
    spec = WORKFLOWS[name]
    module, function = spec.target.split(":")
    return getattr(importlib.import_module("llmbias." + module), function)


def read_frame(path):
    import pandas as pd
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    return pd.read_csv(path)


def save_result(result, path):
    """New output interface; never overwrites historical artifacts."""
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, pd.DataFrame):
        with path.open("x") as stream:
            result.to_csv(stream, index=False)
    else:
        def encode(value):
            if isinstance(value, pd.DataFrame):
                return value.to_dict(orient="records")
            if hasattr(value, "item"):
                return value.item()
            raise TypeError(f"Cannot serialize {type(value).__name__}")
        with path.open("x") as stream:
            json.dump(result, stream, indent=2, default=encode)
            stream.write("\n")


def run(name, configuration, output=None):
    spec = WORKFLOWS[name]
    options = dict(configuration)
    target = resolve(name)
    bound = inspect.signature(target).bind(**options)
    bound.apply_defaults()
    options = dict(bound.arguments)
    # Guard the newly exposed notebook entry points. Scientific functions remain
    # directly callable with their historical behavior for equivalence tests.
    file_keys = {"output_path", "output_csv_path", "out_path", "out_group_csv", "out_examples"}
    for key, value in options.items():
        if key in file_keys and value:
            path = Path(value)
            if path.exists():
                raise FileExistsError(f"Select a fresh output path: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
        if key == "output_dir" and value:
            path = Path(value)
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(f"Select an empty output directory: {path}")
    if output and Path(output).exists():
        raise FileExistsError(f"Select a fresh output path: {output}")
    for key in spec.frames:
        if key in options:
            options[key] = read_frame(options[key])
    for key in spec.frame_lists:
        if key in options:
            options[key] = [read_frame(path) for path in options[key]]
    result = target(**options)
    if output is not None:
        save_result(result, output)
    return result
