import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from spatial_memory.depth_prior import multi_view_depth_masks, reliable_depth_mask

parser = argparse.ArgumentParser()
parser.add_argument('geometry_npz', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
if args.output.exists(): raise ValueError('Choose a new output file')
with np.load(args.geometry_npz, allow_pickle=False) as src:
    depth, confidence, e, k = [src[key] for key in ('depth','confidence','extrinsics','intrinsics')]
if depth.ndim == 4: depth = depth[..., 0]
if confidence.ndim == 4: confidence = confidence[..., 0]
masks, counts = multi_view_depth_masks(depth, confidence, e, k)
digest = hashlib.sha256(args.geometry_npz.read_bytes()).hexdigest()
np.savez_compressed(args.output, mask=masks, support=counts, source_npz_sha256=digest)
report = dict(source_npz_sha256=digest, training_only=True, tolerance=.03, minimum_other_views=2,
              mask_pixels=int(masks.sum()), baseline_pixels=int(sum(reliable_depth_mask(d,c).sum() for i,(d,c) in enumerate(zip(depth,confidence)) if i%8)),
              per_frame=[dict(frame=i, pixels=int(m.sum())) for i,m in enumerate(masks)])
args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='per_frame'},indent=2))
