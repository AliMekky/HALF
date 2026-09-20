"""Check the paper-only registry and its standalone filesystem configuration."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from llmbias.data.paths import data_root, project_root
from llmbias.data.registry import functions
from llmbias._workflows import WORKFLOWS
from llmbias.evaluation.summarization.scoring import commands


class ReleaseTests(unittest.TestCase):
    def test_manifest_covers_exactly_the_paper_datasets(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / 'data/datasets-manifest.json').read_text())
        inputs = {Path(row['destination']).stem for row in manifest['files']
                  if row['role'] == 'paper-input'}
        self.assertEqual(inputs, set(functions))
        self.assertTrue(all(row['destination'].startswith('data/processed/')
                            for row in manifest['files']))

    def test_paths_do_not_fall_back_to_the_old_repository(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(data_root(), project_root() / 'data/processed')
            root, _ = commands('input.jsonl', 'output')
            self.assertEqual(root, (project_root() / 'third_party/summary_bias').resolve())

    def test_only_paper_workflows_are_registered(self):
        for removed in ('evaluate-recruitment-cell304', 'evaluate-recommendation-cell379',
                        'evaluate-bold-reference', 'evaluate-bold-reference-sentiment',
                        'convert-bold-reference'):
            self.assertNotIn(removed, WORKFLOWS)
        self.assertFalse(any(spec.target.startswith('historical.')
                             for spec in WORKFLOWS.values()))


if __name__ == '__main__':
    unittest.main()
