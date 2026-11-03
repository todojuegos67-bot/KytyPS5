"""Package locally rebuilt prompt artwork; never include original textures."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--originals', type=Path, required=True)
parser.add_argument('--converted', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
output = root / 'launcher-windows/KeyboardPrompts'
names = sorted(json.loads((Path(__file__).parent / 'labels.json').read_text(encoding='utf-8')))
rows, packed = [], {}
for name in names:
    filename = name + '.ctxr'
    original = (args.originals / filename).read_bytes()
    converted = (args.converted / filename).read_bytes()
    if original == converted:
        raise ValueError(f'{filename}: converted artwork must differ from the original')
    digest = lambda data: hashlib.sha256(data).hexdigest()
    rows.append(f'{filename} {digest(original)} {digest(converted)}')
    packed[filename + '.gz'] = gzip.compress(converted, mtime=0)
if len(rows) != 22:
    raise ValueError('Expected 22 icons')
output.mkdir(exist_ok=True)
for name, data in packed.items():
    (output / name).write_bytes(data)
(output / 'manifest.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
print('Packed 22 generated textures; no original textures copied.')
