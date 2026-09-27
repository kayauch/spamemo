"""Add derived hybrid assets without rebuilding or overwriting raw assets."""
import argparse
import hashlib
import json
from pathlib import Path
from spatial_memory.hybrid import build_hybrid

parser = argparse.ArgumentParser()
parser.add_argument('manifest', type=Path)
args = parser.parse_args()
path = args.manifest.resolve()
manifest = json.loads(path.read_text(encoding='utf-8'))
source = Path(manifest['source_point_cloud'])
shell = Path(manifest['shell']['path'])
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
before = {str(p): digest(p) for p in (source, shell)}
backup = path.with_name('manifest.before-hybrid.json')
if not backup.exists():
    backup.write_bytes(path.read_bytes())
manifest['hybrid'] = build_hybrid(source, shell, manifest['components'],
    manifest['structural_layout'], manifest['metrics']['scene_diagonal'], path.parent)
assert before == {str(p): digest(p) for p in (source, shell)}
manifest['hybrid']['raw_sha256'] = before
temporary = path.with_suffix('.hybrid.tmp')
temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
temporary.replace(path)
print(json.dumps(manifest['hybrid'], indent=2))
