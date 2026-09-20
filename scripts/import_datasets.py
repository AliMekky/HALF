"""Copy only the paper's manifest-listed inputs, verifying original checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Original LLMBias repository')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'data/datasets-manifest.json').read_text())
    copies = []
    for entry in manifest['files']:
        source = args.source / entry['source']
        destination = root / entry['destination']
        if not source.is_file() or sha256(source) != entry['sha256']:
            parser.error(f'Missing or changed source input: {source}')
        if destination.exists() and sha256(destination) != entry['sha256']:
            parser.error(f'Refusing to overwrite different content: {destination}')
        copies.append((source, destination, entry['sha256']))
    for source, destination, expected in copies:
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if sha256(destination) != expected:
            raise RuntimeError(f'Checksum mismatch after copying: {destination}')
    print(f'Verified {len(copies)} paper input files in {root / "data/processed"}')


if __name__ == '__main__':
    main()
