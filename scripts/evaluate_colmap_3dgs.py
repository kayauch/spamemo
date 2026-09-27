"""Render held-out images, without VGGT priors or modifying any existing scene."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'third_party/3dgs'))
import numpy as np
import torch
from PIL import Image, ImageDraw
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim

parser = argparse.ArgumentParser()
model_args = ModelParams(parser)
pipe_args = PipelineParams(parser)
args = parser.parse_args()
model = model_args.extract(args)
output = Path(model.model_path)/'evaluation'
output.mkdir(exist_ok=False)
gaussians = GaussianModel(model.sh_degree)
scene = Scene(model, gaussians, load_iteration=7000, shuffle=False)
records, panels = [], []
started = time.perf_counter()
with torch.no_grad():
    for camera in scene.getTestCameras():
        rendered = render(camera, gaussians, pipe_args.extract(args), torch.zeros(3, device='cuda'))['render'].clamp(0, 1)
        truth = camera.original_image.cuda()
        score = float(psnr(rendered, truth).mean())
        structural = float(ssim(rendered, truth))
        if not np.isfinite(score + structural):
            raise ValueError('Non-finite validation metric')
        records.append({'image': camera.image_name, 'psnr': score, 'ssim': structural})
        pair = []
        for tensor in (truth, rendered):
            image = Image.fromarray((tensor.permute(1, 2, 0).cpu().numpy()*255).astype('uint8'))
            image.thumbnail((400, 300))
            pair.append(image)
        panel = Image.new('RGB', (800, 330), '#222222')
        panel.paste(pair[0], (0, 30)); panel.paste(pair[1], (400, 30))
        ImageDraw.Draw(panel).text((5, 8), f'{camera.image_name} | source / 3DGS | PSNR {score:.2f} SSIM {structural:.3f}', fill='white')
        panel.save(output/f'{camera.image_name}.jpg')
        panels.append(panel)
if not records:
    raise ValueError('No held-out cameras')
sheet = Image.new('RGB', (800, 330*min(6, len(panels))), '#222222')
for index, panel in enumerate(panels[:6]):
    sheet.paste(panel, (0, 330*index))
sheet.save(output/'comparison.jpg')
ply = Path(model.model_path)/'point_cloud/iteration_7000/point_cloud.ply'
report = {'method': 'COLMAP + official 3DGS; no VGGT', 'held_out_views': records,
          'mean_psnr': float(np.mean([r['psnr'] for r in records])),
          'mean_ssim': float(np.mean([r['ssim'] for r in records])),
          'model_sha256': hashlib.sha256(ply.read_bytes()).hexdigest(),
          'evaluation_seconds': time.perf_counter()-started,
          'limitations': 'Held-out appearance check only; geometry and unseen views need visual review.'}
(output/'report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
