"""Offline equivalence checks; deliberately retain historical scientific behavior."""
import ast
import contextlib
import copy
import importlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook_functions(cell, globals_=None):
    nb = json.loads((LEGACY / "data_analysis.ipynb").read_text())
    tree = ast.parse("".join(nb["cells"][cell]["source"]))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"pd": pd, "np": np}
    if globals_:
        namespace.update(globals_)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<legacy-cell>", "exec"), namespace)
    return namespace


class NormalizePaths(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.startswith("/Users/alimekky/Documents/Bias Evaluation/"):
            return ast.Constant(value="RELOCATED_PATH")
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "str" and len(node.args) == 1:
            inner = node.args[0]
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "workspace_path":
                return ast.Constant(value="RELOCATED_PATH")
        return self.generic_visit(node)


class NormalizeBatchScope(NormalizePaths):
    """Allow only the explicit task-removal/registry guard changes in the driver."""
    def visit_If(self, node):
        if ast.unparse(node.test) == 'dataset not in functions':
            return None
        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        node.values = [v for v in node.values if ast.unparse(v) != "dataset == 'dreaddit'"]
        return self.generic_visit(node)


@unittest.skipUnless(LEGACY.exists(), "Restore legacy artifact bundle first")
class PreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts = load_file(LEGACY / "experiments/prompts.py", "legacy_prompts")
        previous = sys.modules.get("prompts")
        sys.modules["prompts"] = cls.prompts
        try:
            cls.utils = load_file(LEGACY / "experiments/utils.py", "legacy_utils")
        finally:
            if previous is None:
                del sys.modules["prompts"]
            else:
                sys.modules["prompts"] = previous

    def test_extracted_function_bodies(self):
        mappings = json.loads((ROOT / "docs/module-map.json").read_text())
        nb = json.loads((LEGACY / "data_analysis.ipynb").read_text())
        for entry in mappings:
            if "symbol" not in entry:
                continue
            with self.subTest(symbol=entry["symbol"], target=entry["target"]):
                original = "".join(nb["cells"][entry["cell"]]["source"]) if "cell" in entry else (LEGACY / entry["source"]).read_text()
                moved = (ROOT / entry["target"]).read_text()
                a = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == entry["symbol"])
                b = next(n for n in ast.parse(moved).body if isinstance(n, ast.FunctionDef) and n.name == entry["symbol"])
                self.assertEqual(ast.dump(NormalizePaths().visit(a)), ast.dump(NormalizePaths().visit(b)))

    def test_moved_script_function_bodies(self):
        mappings = json.loads((ROOT / "docs/module-map.json").read_text())
        for entry in mappings:
            if "symbol" in entry:
                continue
            original = ast.parse((LEGACY / entry["source"]).read_text())
            moved = ast.parse((ROOT / entry["target"]).read_text())
            after = {n.name: n for n in moved.body if isinstance(n, ast.FunctionDef)}
            for node in original.body:
                if isinstance(node, ast.FunctionDef):
                    with self.subTest(source=entry["source"], function=node.name):
                        normalizer = NormalizeBatchScope if entry["source"] == "experiments/create_batch_file.py" else NormalizePaths
                        self.assertEqual(ast.dump(normalizer().visit(node)), ast.dump(normalizer().visit(after[node.name])))

    def test_prompt_constants(self):
        from llmbias import prompts
        for name in vars(self.prompts):
            if name.isupper():
                if name == "DATASETS":
                    expected = {k: v for k, v in self.prompts.DATASETS.items() if k not in {"dreaddit", "education_ranking_generation"}}
                    self.assertEqual(expected, prompts.DATASETS)
                elif name != "SYSTEM_PROMPT_EDUCATION_ADMISSION":
                    self.assertEqual(getattr(self.prompts, name), getattr(prompts, name), name)

    def test_all_registered_builders_on_saved_data(self):
        from llmbias.registry import functions
        paths = {
            "CAMS": "mental_health_data/CAMS", "SAD": "mental_health_data/SAD",
            "dreaddit": "mental_health_data/dreaddit", "medbullets": "medical_data/medbullets",
            "medical_bias": "medical_data/medical_bias", "movielens": "recommendation_system/movielens",
            "djinni": "admission_data/djinni", "education_ranking": "education_data/education_ranking",
            "education_ga": "education_data/education_ga", "mt_gender": "translation_data/mt_gender",
            "ecthr": "legal_data/ecthr", "ontonotes": "summarization_data/ontonotes",
            "bold": "conv_ai/BOLD", "bbq": "conv_ai/bbq",
        }
        self.assertEqual(set(functions), set(self.utils.functions) - {"dreaddit", "education_ga"})
        for name, path in paths.items():
            if name not in functions:
                continue
            with self.subTest(dataset=name):
                df = pd.read_csv(LEGACY / (path + ".csv")).head(3)
                template = {"custom_id": None, "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "temperature": 0.6, "messages": [{"role": "system", "content": None}, {"role": "user", "content": None}]}}
                random.seed(123)
                original = self.utils.functions[name](df, "gpt-4o", name, copy.deepcopy(template))
                random.seed(123)
                moved = functions[name](df, "gpt-4o", name, copy.deepcopy(template))
                self.assertEqual(original, moved)

    def test_new_build_wrapper_matches_legacy_cli(self):
        from llmbias.cli import build
        import os
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "medical_data"
            data.mkdir()
            pd.read_csv(LEGACY / "medical_data/medbullets.csv").head(3).to_csv(data / "medbullets.csv", index=False)
            cwd, argv = Path.cwd(), sys.argv
            old_modules = {name: sys.modules.get(name) for name in ["prompts", "utils"]}
            sys.modules.update(prompts=self.prompts, utils=self.utils)
            try:
                script = load_file(LEGACY / "experiments/create_batch_file.py", "legacy_batch")
                for model in ["gpt-4o", "o4-mini-2025-04-16"]:
                    working = root / "experiments"
                    expected = working / "medical_data/batch_files"
                    expected.mkdir(parents=True, exist_ok=True)
                    sys.argv = ["create_batch_file", "--dataset", "medical_data/medbullets", "--model", model]
                    os.chdir(working)
                    with contextlib.redirect_stdout(io.StringIO()):
                        script.main()
                        build("medical_data/medbullets", model, root / model, root)
                    for path in expected.glob(model + "_*.jsonl"):
                        self.assertEqual(path.read_bytes(), (root / model / path.name).read_bytes())
            finally:
                os.chdir(cwd)
                sys.argv = argv
                for name, value in old_modules.items():
                    if value is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = value

    def test_parsers(self):
        for original, new in [("extract_mcq_answers_openai.py", "openai_mcq"), ("extract_data_claude.py", "claude_mcq"), ("extract_mcq_answer_deepseek.py", "deepseek_mcq")]:
            before = load_file(LEGACY / "experiments" / original, "legacy_" + new)
            after = importlib.import_module("llmbias.parsing." + new)
            if new == "openai_mcq":
                entry = {"custom_id": "request-3-gpt-4o-medbullets-arab-female", "response": {"body": {"choices": [{"message": {"content": " A "}}]}}}
                self.assertEqual(before.extract_metadata(entry), after.extract_metadata(entry))
            elif new == "claude_mcq":
                self.assertEqual(before.parse_custom_id("request-3-claude-medbullets-arab-female"), after.parse_custom_id("request-3-claude-medbullets-arab-female"))
            else:
                self.assertEqual(before.group_conversations({1: "A", 2: "B"}), after.group_conversations({1: "A", 2: "B"}))

    def test_recruitment_analysis(self):
        from llmbias.analysis import recruitment
        before = notebook_functions(306)
        neutral = pd.DataFrame({"index": [0, 1], "decision": ["Admit", "Reject"]})
        sensitive = pd.DataFrame({"index": [0, 1], "decision": ["Reject", "Reject"], "gender": ["female", "female"], "ethnicity": ["arab", "arab"]})
        a = before["add_reference_columns"](neutral, sensitive)
        b = recruitment.add_reference_columns(neutral, sensitive)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(before["summarise"](a, ["gender"], "gender"), recruitment.summarise(b, ["gender"], "gender"))

    def test_recommendation_analysis(self):
        from llmbias.analysis import recommendation
        before = notebook_functions(382)
        for a, b in [(["A", "B", "C"], ["B", "A", "D"]), ([], []), (["A"], ["A"])]:
            for name in ["jaccard", "prag"]:
                self.assertEqual(before[name](a, b), getattr(recommendation, name)(a, b))

    def test_classification_metrics(self):
        from sklearn.metrics import f1_score, confusion_matrix
        from llmbias.analysis import cams, sad
        before = notebook_functions(338, {"f1_score": f1_score, "confusion_matrix": confusion_matrix})
        df = pd.DataFrame({"group": ["a", "a", "b", "b"], "gold": [0, 1, 0, 1], "pred": [0, 1, 1, 1]})
        self.assertEqual(before["equalized_odds_gap"](df, "group", "gold", "pred"), cams.equalized_odds_gap(df, "group", "gold", "pred"))
        before = notebook_functions(358, {"f1_score": f1_score})
        gold, pred = [[1, 0], [1, 1]], [[0, 1], [1, 1]]
        self.assertEqual(before["compute_multilabel_f1"](gold, pred), sad.compute_multilabel_f1(gold, pred))


if __name__ == "__main__":
    unittest.main()
