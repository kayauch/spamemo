"""Appearance-only distillation of three geometry-checked AI repair views.

All Gaussian centers, opacity, scale and rotation remain frozen. Non-repair
pixels use the previous best model as a teacher. Held-out photos are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/2dgs'))
import numpy as np
import torch
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss,ssim

p=argparse.ArgumentParser(); mp,pp,op=ModelParams(p),PipelineParams(p),OptimizationParams(p)
p.add_argument('--initial-ply',type=Path,required=True)
p.add_argument('--control-teacher',type=Path,help='Ablation: use unchanged teacher pixels instead of AI patches')
a=p.parse_args()
output=Path(a.model_path)
if output.exists(): raise FileExistsError(output)
repair=json.loads((Path(a.source_path)/'repair.json').read_text())
if not repair['consistency']['passed']: raise ValueError('Multi-view checks failed')
if hashlib.sha256(a.initial_ply.read_bytes()).hexdigest()!=repair['source_ply_sha256']: raise ValueError('Best model changed')
if not a.eval or a.resolution!=1: raise ValueError('Use fixed split and native resolution')
np.random.seed(42); random.seed(42); torch.manual_seed(42)
dataset,pipe,opt=mp.extract(a),pp.extract(a),op.extract(a)
output.mkdir(parents=True)
model=GaussianModel(dataset.sh_degree); scene=Scene(dataset,model,shuffle=False)
model.load_ply(str(a.initial_ply)); model.spatial_lr_scale=scene.cameras_extent
opt.feature_lr=.0003
model.training_setup(opt)
frozen={name:getattr(model,name).detach().clone() for name in ('_xyz','_scaling','_rotation','_opacity')}
for name in frozen: getattr(model,name).requires_grad_(False)
changed={r['frame'] for r in repair['consistency']['changed_views']}
control_images={}
if a.control_teacher:
    from PIL import Image
    for frame in changed:
        array=np.array(Image.open(a.control_teacher/frame/'render.png').convert('RGB'))
        control_images[frame]=torch.tensor(array.transpose(2,0,1)/255.,dtype=torch.float32,device='cuda')
if changed & {c.image_name for c in scene.getTestCameras()}: raise ValueError('Held-out repair leakage')
(output/'cfg_args').write_text(str(argparse.Namespace(**vars(dataset))))
history=[]; stack=[]; started=time.monotonic()
for step in range(1,opt.iterations+1):
    if not stack: stack=scene.getTrainCameras().copy(); random.shuffle(stack)
    camera=stack.pop()
    package=render(camera,model,pipe,torch.zeros(3,device='cuda'))
    gt=control_images.get(camera.image_name)
    if gt is None: gt=camera.original_image.cuda()
    loss=.8*l1_loss(package['render'],gt)+.2*(1-ssim(package['render'],gt))
    loss.backward(); model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss): raise ValueError('Nonfinite repair loss')
    if step==1 or step%250==0:
        row=dict(step=step,loss=float(loss.detach()),seconds=round(time.monotonic()-started,1))
        history.append(row); print(json.dumps(row),flush=True)
if not all(torch.equal(saved,getattr(model,name)) for name,saved in frozen.items()):
    raise ValueError('Frozen geometry or opacity changed')
scene.save(opt.iterations)
record=dict(iterations=opt.iterations,feature_lr=opt.feature_lr,seed=42,appearance_only=True,
            no_ai_control=bool(a.control_teacher),
            geometry_and_opacity_bit_exact=True,source_ply_sha256=repair['source_ply_sha256'],
            changed_views=sorted(changed),held_out_views=[c.image_name for c in scene.getTestCameras()],
            caveat='Teacher pixels are preserved as targets, not guaranteed unchanged in final render; evaluate before adoption')
(output/'repair-training.json').write_text(json.dumps(record,indent=2))
(output/'training-history.json').write_text(json.dumps(history,indent=2))
