"""Run with the isolated WSL Python after official 2DGS training."""
import argparse
import json
import hashlib
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'third_party/2dgs'))
sys.path.insert(0, str(root / 'src'))
from spatial_memory.depth_prior import reliable_depth_mask
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.loss_utils import ssim

parser = argparse.ArgumentParser()
model_args = ModelParams(parser)
pipe_args = PipelineParams(parser)
parser.add_argument('--iteration', type=int, default=7000)
parser.add_argument('--geometry-npz', type=Path)
parser.add_argument('--output-name', default='evaluation')
parser.add_argument('--output-dir', type=Path, help='New external report directory; leaves model directory untouched')
parser.add_argument('--repair-reference', type=Path, help='Frozen best snapshot; measure depth drift and holes at matching cameras')
parser.add_argument('--repair-spec', type=Path, help='Repair COLMAP directory for rendered multi-view patch consistency')
args = parser.parse_args()
dataset = model_args.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
pipe = pipe_args.extract(args)
if Path(args.output_name).name != args.output_name:
    raise ValueError('Output name must be a single directory name')
out = args.output_dir or (Path(args.model_path) / args.output_name)
out.mkdir(parents=True, exist_ok=args.output_dir is None)
records, panels = [], []
centers_error = []
priors = None
alignment = json.loads((Path(args.source_path)/'alignment.json').read_text())
if args.geometry_npz and alignment.get('cameras_fixed', True):
    with np.load(args.geometry_npz, allow_pickle=False) as saved:
        priors = [saved['depth'], saved['confidence']]
    priors = [a[..., 0] if a.ndim == 4 else a for a in priors]
with torch.no_grad():
    for camera in scene.getTestCameras():
        package = render(camera, gaussians, pipe, torch.zeros(3, device='cuda'))
        rendered = package['render'].clamp(0,1)
        reference = camera.original_image.cuda()
        score = float(psnr(rendered, reference).mean())
        structure = float(ssim(rendered, reference).mean())
        if not np.isfinite(score + structure):
            raise ValueError('Nonfinite rendered validation score')
        records.append({'frame':camera.image_name, 'psnr':score, 'ssim':structure})
        if args.repair_reference:
            with np.load(args.repair_reference/camera.image_name/'geometry.npz',allow_pickle=False) as reference_depth:
                old_depth=reference_depth['depth']
                old_alpha=reference_depth['alpha']
            new_depth=package['surf_depth'][0].cpu().numpy()
            new_alpha=package['rend_alpha'][0].cpu().numpy()
            valid=np.isfinite(old_depth)&(old_depth>0)&(old_alpha>.5)
            if new_depth.shape!=old_depth.shape or not valid.any(): raise ValueError('Invalid repair reference')
            records[-1].update(depth_to_best_relative=float(np.mean(np.abs(new_depth[valid]-old_depth[valid])/old_depth[valid])),
                               hole_fraction=float(np.mean(new_alpha<.5)))
        if priors is not None:
            index = int(camera.image_name)
            d, c = priors[0][index], priors[1][index]
            mask = reliable_depth_mask(d, c)
            prediction = package['surf_depth'][0].cpu().numpy()
            alpha = package['rend_alpha'][0].cpu().numpy()
            if prediction.shape != d.shape:
                # Evaluate every candidate on exactly the original VGGT mask/grid.
                prediction = cv2.resize(prediction, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
                alpha = cv2.resize(alpha, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
            if not mask.any():
                raise ValueError('Empty depth evaluation mask')
            error = np.abs(prediction[mask] - d[mask]) / d[mask]
            records[-1].update(depth_abs_relative=float(error.mean()),
                               depth_p90_relative=float(np.quantile(error, .9)),
                               depth_mask_pixels=int(mask.sum()),
                               alpha_coverage=float((alpha[mask] > .5).mean()))
        array = (rendered.permute(1,2,0).cpu().numpy()*255).astype('uint8')
        photo = Image.fromarray(array)
        photo.save(out/f'{camera.image_name}.png')
        if len(panels) < 4:
            gt = Image.fromarray((reference.permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            panel = Image.new('RGB',(photo.width*2,photo.height+26),'#202020')
            panel.paste(gt,(0,26)); panel.paste(photo,(photo.width,26))
            ImageDraw.Draw(panel).text((8,6),f'{camera.image_name}: video (left) / 2DGS (right), PSNR {score:.2f}',fill='white')
            panels.append(panel)
    for camera in scene.getTrainCameras()+scene.getTestCameras():
        expected = -np.asarray(camera.R) @ np.asarray(camera.T)
        actual = camera.camera_center.cpu().numpy()
        centers_error.append(float(np.linalg.norm(expected-actual)))
if not records:
    raise ValueError('No held-out views: train and evaluate with --eval')
board=Image.new('RGB',(panels[0].width,sum(p.height for p in panels)))
y=0
for panel in panels:
    board.paste(panel,(0,y)); y+=panel.height
board.save(out/'comparison.jpg',quality=95)
report={'iteration':args.iteration, 'splats':len(gaussians.get_xyz),
        'alignment_sha256':hashlib.sha256((Path(args.source_path)/'alignment.json').read_bytes()).hexdigest(),
        'camera_files_sha256':hashlib.sha256((Path(args.source_path)/'sparse/0/cameras.txt').read_bytes()+(Path(args.source_path)/'sparse/0/images.txt').read_bytes()).hexdigest(),
        'image_size': [scene.getTestCameras()[0].image_width, scene.getTestCameras()[0].image_height],
        'model_ply_sha256':hashlib.sha256((Path(args.model_path)/f'point_cloud/iteration_{args.iteration}/point_cloud.ply').read_bytes()).hexdigest(),
        'test_views':records, 'mean_psnr':float(np.mean([r['psnr'] for r in records])),
        'mean_ssim':float(np.mean([r['ssim'] for r in records])),
        'max_camera_center_error':max(centers_error),
        'world_from_2dgs':np.eye(4).tolist(),
        'validation_scope':'Every eighth frame excluded from appearance training; VGGT geometry used all frames.',
        'finite_parameters':all(bool(torch.isfinite(t).all()) for t in (gaussians.get_xyz,gaussians.get_scaling,gaussians.get_rotation,gaussians.get_opacity))}
if not report['finite_parameters']:
    raise ValueError('Nonfinite splat parameters')
if args.repair_spec:
    from spatial_memory.scene_repair import correspondence,sample
    if not args.repair_reference: raise ValueError('Repair consistency needs a reference snapshot')
    spec=json.loads((args.repair_spec/'repair.json').read_text())
    selected={spec['anchor'],*spec['targets']}
    views={}
    repair_out=out/'repair-views'; repair_out.mkdir()
    with torch.no_grad():
        for camera in scene.getTrainCameras():
            if camera.image_name not in selected: continue
            package=render(camera,gaussians,pipe,torch.zeros(3,device='cuda'))
            rgb=package['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
            Image.fromarray((rgb*255).astype(np.uint8)).save(repair_out/f'{camera.image_name}.png')
            with np.load(args.repair_reference/camera.image_name/'geometry.npz',allow_pickle=False) as saved:
                pose,intrinsic=saved['extrinsic'],saved['intrinsic']
            views[camera.image_name]=(rgb,package['surf_depth'][0].cpu().numpy(),pose,intrinsic)
    source=views[spec['anchor']]; errors=[]; counts=[]
    for frame in spec['targets']:
        target=views[frame]
        uv,valid=correspondence(target[1],target[2],target[3],source[1],source[2],source[3])
        approved=np.asarray(Image.open(args.repair_spec/'provenance'/f'{frame}.png'))==3
        approved &= valid
        if not approved.any(): raise ValueError('Repair patch lost multi-view support')
        projected=sample(source[0],uv)
        errors.append(float(np.abs(target[0][approved]-projected[approved]).mean()))
        counts.append(int(approved.sum()))
    report['repair_multiview_mae']=float(np.mean(errors))
    report['repair_multiview_pixels']=counts
    report['repair_multiview_scope']='Rendered RGB consistency on repaired training regions; not independent geometry ground truth'
if args.repair_reference:
    reference_meta=json.loads((args.repair_reference/'snapshot.json').read_text())
    if report['alignment_sha256']!=reference_meta['alignment_sha256']: raise ValueError('Repair reference uses different cameras')
    report['repair_reference_model_sha256']=reference_meta['source_ply_sha256']
    report['mean_depth_to_best_relative']=float(np.mean([r['depth_to_best_relative'] for r in records]))
    report['mean_hole_fraction']=float(np.mean([r['hole_fraction'] for r in records]))
if priors is not None:
    report['geometry_npz_sha256'] = hashlib.sha256(args.geometry_npz.read_bytes()).hexdigest()
    report['mean_depth_abs_relative'] = float(np.mean([r['depth_abs_relative'] for r in records]))
    report['mean_alpha_coverage'] = float(np.mean([r['alpha_coverage'] for r in records]))
    report['depth_reference'] = 'VGGT estimated depth, not ground truth; same fixed confidence/edge masks across candidates'
elif not alignment.get('cameras_fixed', True):
    report['depth_evaluation_skipped'] = 'Corrected appearance cameras do not match original per-pixel VGGT depth grid'
    report['camera_calibration_scope'] = alignment['appearance_camera_refinement']['camera_calibration_scope']
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
