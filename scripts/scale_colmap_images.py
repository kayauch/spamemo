"""Use original video detail at higher resolution without moving VGGT cameras/points."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from spatial_memory.texturing import preprocess_frame

parser = argparse.ArgumentParser()
parser.add_argument('scene_dir', type=Path)
parser.add_argument('source_colmap', type=Path)
parser.add_argument('output_colmap', type=Path)
parser.add_argument('--scale', type=int, default=2)
args = parser.parse_args()
if args.scale < 1 or args.output_colmap.exists():
    raise ValueError('Use a positive scale and a new output directory')
alignment = json.loads((args.source_colmap / 'alignment.json').read_text())
digest = hashlib.sha256((args.scene_dir / 'reconstruction/reconstruction.npz').read_bytes()).hexdigest()
if alignment['source_npz_sha256'] != digest:
    raise ValueError('Mismatched geometry and COLMAP dataset')
sparse = args.output_colmap / 'sparse/0'
sparse.mkdir(parents=True)
(args.output_colmap / 'images').mkdir()
source_sparse = args.source_colmap / 'sparse/0'
lines = []
for line in (source_sparse / 'cameras.txt').read_text().splitlines():
    if not line or line.startswith('#'):
        lines.append(line); continue
    parts = line.split()
    if parts[1] != 'PINHOLE': raise ValueError('Only PINHOLE cameras supported')
    parts[2:4] = [str(int(x)*args.scale) for x in parts[2:4]]
    parts[4:] = [str(float(x)*args.scale) for x in parts[4:]]
    lines.append(' '.join(parts))
(sparse / 'cameras.txt').write_text('\n'.join(lines)+'\n')
lines, observation = [], False
for line in (source_sparse / 'images.txt').read_text().splitlines():
    if line.startswith('#'):
        lines.append(line); continue
    if observation:
        parts = line.split()
        for j in range(0, len(parts), 3):
            parts[j], parts[j+1] = str(float(parts[j])*args.scale), str(float(parts[j+1])*args.scale)
        lines.append(' '.join(parts))
    else:
        lines.append(line)
    observation = not observation
(sparse / 'images.txt').write_text('\n'.join(lines)+'\n')
for name in ('points3D.txt', 'points3D.ply'):
    if (source_sparse/name).exists(): shutil.copy2(source_sparse/name, sparse/name)
records = json.loads((args.scene_dir/'frames/manifest.json').read_text())['frames']
w, h = alignment['image_size']
for index, record in enumerate(records):
    preprocess_frame(args.scene_dir/'frames'/record['name'], h, w, args.scale).save(args.output_colmap/'images'/f'{index:06d}.png')
alignment['image_size'] = [w*args.scale, h*args.scale]
alignment['appearance_image_scale'] = args.scale
alignment['appearance_source'] = 'Original video frames resized directly, without upscaling the 518px images'
(args.output_colmap/'alignment.json').write_text(json.dumps(alignment, indent=2))
print(json.dumps(alignment, indent=2))
