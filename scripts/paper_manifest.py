"""Validate or inspect paper recipes without executing experiments or calling APIs."""
import argparse
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def validate(manifest):
    from llmbias._workflows import resolve
    from llmbias.cli import COMMANDS
    from llmbias.data.registry import functions

    if set(manifest['recipes']) != set(functions):
        raise ValueError('Recipes must cover exactly the active paper datasets')
    if {t['id'] for t in manifest['tables']} != {f'table_{i:02d}' for i in range(1, 16)}:
        raise ValueError('Expected Tables 1 through 15')
    models = {model['id'] for model in manifest['models']}
    artifacts = {artifact['id'] for artifact in manifest['artifacts']}
    for dataset, recipe in manifest['recipes'].items():
        if {run['model_id'] for run in recipe['runs']} != models:
            raise ValueError(f'Incomplete model inventory: {dataset}')
        for run in recipe['runs']:
            if not set(run['candidate_artifact_ids']) <= artifacts:
                raise ValueError(f'Unknown artifact reference: {dataset}')
        for step in recipe['steps']:
            if step['kind'] == 'workflow':
                config = json.loads((ROOT / step['config']).read_text())
                inspect.signature(resolve(step['name'])).bind(**config)
            elif step['kind'] == 'script':
                if step['name'] not in COMMANDS:
                    raise ValueError(f'Unknown script: {step["name"]}')
            else:
                raise ValueError(f'Unknown step kind: {step["kind"]}')
    return (f'Valid structure: {len(manifest["tables"])} tables, '
            f'{len(manifest["figures"])} figures, {len(manifest["recipes"])} datasets, '
            f'{sum(len(r["runs"]) for r in manifest["recipes"].values())} run mappings. '
            'This checks references and argument signatures, not paper results or input contents.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'show'])
    parser.add_argument('table', nargs='?', help='For example table_14')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'experiments/paper_manifest.json').read_text())
    if args.action == 'validate':
        print(validate(manifest))
    else:
        table = next((t for t in manifest['tables'] if t['id'] == args.table), None)
        if table is None:
            parser.error('Choose a table ID from table_01 through table_15')
        result = {'table': table}
        if 'recipe' in table:
            result['recipe'] = manifest['recipes'][table['recipe']]
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
