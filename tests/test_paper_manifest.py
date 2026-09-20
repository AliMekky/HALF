"""Keep the paper inventory linked to callable package entry points."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_manifest_tool', ROOT / 'scripts/paper_manifest.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class PaperManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / 'experiments/paper_manifest.json').read_text())

    def test_all_recipes_bind_to_current_workflows(self):
        self.assertIn('96 run mappings', tool.validate(self.manifest))

    def test_unknown_artifact_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['recipes']['bold']['runs'][0]['candidate_artifact_ids'].append('missing')
        with self.assertRaisesRegex(ValueError, 'Unknown artifact'):
            tool.validate(manifest)

    def test_missing_model_entry_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['recipes']['bold']['runs'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete model inventory'):
            tool.validate(manifest)


if __name__ == '__main__':
    unittest.main()
