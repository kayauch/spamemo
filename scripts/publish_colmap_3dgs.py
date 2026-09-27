"""Publish a tested COLMAP/3DGS experiment to the local static viewer only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
run = args.run.resolve()
report = json.loads((run/'report.json').read_text())
model = run/'3dgs-7000'
evaluation = json.loads((model/'evaluation/report.json').read_text())
ply = model/'point_cloud/iteration_7000/point_cloud.ply'
if report['status'] != 'ready' or hashlib.sha256(ply.read_bytes()).hexdigest() != evaluation['model_sha256']:
    raise ValueError('Unvalidated model')
vertices = PlyData.read(ply)['vertex'].data
if not len(vertices) or not all(np.isfinite(vertices[name]).all() for name in vertices.dtype.names):
    raise ValueError('Invalid splats')
cameras = sorted(json.loads((model/'cameras.json').read_text()), key=lambda camera: camera['img_name'])
destination = root/'web/colmap-reviews'/run.name
destination.mkdir(parents=True, exist_ok=False)
shutil.copy2(ply, destination/'model.ply')
shutil.copy2(model/'evaluation/comparison.jpg', destination/'comparison.jpg')
review = {'cameras': cameras, 'registered_images': report['registered_images'],
          'total_images': len(report['frames']), 'evaluation': evaluation,
          'splats': len(vertices), 'units': 'arbitrary COLMAP scale',
          'geometry_source': report['geometry_source']}
(destination/'review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
print(f'http://127.0.0.1:8001/static/colmap-gs.html?run={run.name}')
