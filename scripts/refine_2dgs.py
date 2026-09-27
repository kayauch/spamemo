"""Fine-tune official 2DGS using confidence-gated VGGT depth, without camera changes.

Run with the isolated WSL environment. A new output directory is mandatory.
All cameras used for evaluation stay excluded from appearance/depth training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'third_party/2dgs'), str(ROOT / 'src')]
import numpy as np
import cv2
import torch
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.point_utils import depth_to_normal
from utils.general_utils import build_rotation
from spatial_memory.depth_prior import reliable_depth_mask
from spatial_memory.appearance_prior import diffuse_structure_mask


def main():
    parser = argparse.ArgumentParser()
    mp, pp, op = ModelParams(parser), PipelineParams(parser), OptimizationParams(parser)
    parser.add_argument('--initial-ply', type=Path)
    parser.add_argument('--geometry-npz', type=Path, required=True)
    parser.add_argument('--depth-weight', type=float, default=.1)
    parser.add_argument('--normal-prior-weight', type=float, default=0)
    parser.add_argument('--confidence-photo-weighting', action='store_true')
    parser.add_argument('--depth-mask-file', type=Path)
    parser.add_argument('--freeze-positions', action='store_true')
    parser.add_argument('--structure-sh-penalty', type=float, default=0)
    args = parser.parse_args()
    output = Path(args.model_path)
    if output.exists():
        raise ValueError('Choose a new output directory')
    if not args.eval or args.resolution != 1:
        raise ValueError('Use --eval -r 1 to preserve evaluation split and depth resolution')
    alignment = json.loads((Path(args.source_path) / 'alignment.json').read_text())
    if not alignment.get('cameras_fixed', True) and (args.depth_weight or args.normal_prior_weight):
        raise ValueError('Original per-pixel depth priors cannot supervise corrected camera poses')
    source_hash = hashlib.sha256(args.geometry_npz.read_bytes()).hexdigest()
    if alignment['source_npz_sha256'] != source_hash:
        raise ValueError('Depth and COLMAP export belong to different reconstructions')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    output.mkdir(parents=True)
    dataset, pipe, opt = mp.extract(args), pp.extract(args), op.extract(args)
    (output / 'cfg_args').write_text(str(argparse.Namespace(**vars(dataset))))
    model = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, model, shuffle=False)
    if args.initial_ply:
        model.load_ply(str(args.initial_ply))
    model.spatial_lr_scale = scene.cameras_extent
    # A warm start: gentle shape updates, no new splats and no opacity resets.
    opt.position_lr_init = .000016 if args.initial_ply else .00016
    opt.position_lr_final = .0000016
    opt.position_lr_max_steps = opt.iterations
    if args.freeze_positions:
        if not args.initial_ply: raise ValueError('Frozen positions require a trained initial PLY')
        opt.position_lr_init = opt.position_lr_final = 0
    opt.feature_lr = .0005 if args.initial_ply else .0025
    opt.opacity_lr = .005 if args.initial_ply else .05
    opt.scaling_lr = .001 if args.initial_ply else .005
    opt.rotation_lr = .0002 if args.initial_ply else .001
    model.training_setup(opt)
    diffuse_mask = None
    if args.structure_sh_penalty:
        if not args.initial_ply:
            raise ValueError('Structure appearance prior currently requires fixed splat count')
        manifest_path = args.geometry_npz.parent.parent/'spatial_model/manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        normals = build_rotation(model.get_rotation)[:, :, 2].detach().cpu().numpy()
        chosen = diffuse_structure_mask(model.get_xyz.detach().cpu().numpy(), normals, manifest)
        diffuse_mask = torch.tensor(chosen, device='cuda')
        print(f'Diffuse wall/ceiling splats: {chosen.sum()}', flush=True)
    with np.load(args.geometry_npz, allow_pickle=False) as src:
        depth, confidence = src['depth'], src['confidence']
    if depth.ndim == 4: depth = depth[..., 0]
    if confidence.ndim == 4: confidence = confidence[..., 0]
    masks = None
    if args.depth_mask_file:
        with np.load(args.depth_mask_file, allow_pickle=False) as data:
            if str(data['source_npz_sha256']) != source_hash:
                raise ValueError('Depth masks belong to different geometry')
            masks = data['mask']
        if masks.shape != depth.shape or masks[::8].any():
            raise ValueError('Invalid mask shape or held-out training pixels')
    targets = {}
    for cam in scene.getTrainCameras():
        index = int(cam.image_name)
        d, c = depth[index], confidence[index]
        if d.shape != (cam.image_height, cam.image_width):
            scale = alignment.get('appearance_image_scale', 1)
            if (d.shape[0]*scale, d.shape[1]*scale) != (cam.image_height, cam.image_width):
                raise ValueError('Depth and image resolution differ without matching scale metadata')
            d = cv2.resize(d, (cam.image_width, cam.image_height), interpolation=cv2.INTER_LINEAR)
            c = cv2.resize(c, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)
        mask = reliable_depth_mask(d, c)
        if masks is not None:
            extra_mask = masks[index]
            if extra_mask.shape != d.shape:
                extra_mask = cv2.resize(extra_mask.astype(np.uint8), (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask &= extra_mask
        target = torch.tensor(np.where(np.isfinite(d) & (d > 0), d, 1), device='cuda')
        target_mask = torch.tensor(mask, device='cuda')
        target_normal = depth_to_normal(cam, target[None]).permute(2, 0, 1).detach()
        normal_mask = mask.copy()
        for axis in (0, 1):
            normal_mask &= np.roll(mask, 1, axis) & np.roll(mask, -1, axis)
        targets[cam.image_name] = (target, target_mask, target_normal, torch.tensor(normal_mask, device='cuda'))
    config = {'initial_ply': str(args.initial_ply) if args.initial_ply else None, 'source_npz_sha256': source_hash,
              'iterations': opt.iterations, 'depth_weight': args.depth_weight,
              'normal_weight': opt.lambda_normal, 'train_frames': sorted(targets),
              'normal_prior_weight': args.normal_prior_weight,
              'appearance_image_scale': alignment.get('appearance_image_scale', 1),
              'depth_mask_file': str(args.depth_mask_file) if args.depth_mask_file else None,
              'test_frames': [c.image_name for c in scene.getTestCameras()],
              'fixed_cameras': alignment.get('cameras_fixed',True), 'densification': not bool(args.initial_ply), 'seed': 42,
              'freeze_positions': args.freeze_positions,
              'structure_sh_penalty': args.structure_sh_penalty,
              'diffuse_structure_splats': int(diffuse_mask.sum()) if diffuse_mask is not None else 0,
              'confidence_photo_weighting': args.confidence_photo_weighting,
              'depth_mask': 'upper confidence half; exclude >8% depth jumps; alpha >0.5'}
    (output / 'refinement.json').write_text(json.dumps(config, indent=2))
    if alignment.get('appearance_camera_refinement'):
        (output / 'appearance-camera-refinement.json').write_text(json.dumps(alignment['appearance_camera_refinement'],indent=2))
    background = torch.zeros(3, device='cuda')
    stack, history = [], []
    started = time.monotonic()
    for step in range(1, opt.iterations + 1):
        model.update_learning_rate(step)
        if step % 1000 == 0:
            model.oneupSHdegree()
        if not stack:
            stack = scene.getTrainCameras().copy(); random.shuffle(stack)
        camera = stack.pop()
        pkg = render(camera, model, pipe, background)
        gt = camera.original_image.cuda()
        photo = .8 * l1_loss(pkg['render'], gt) + .2 * (1 - ssim(pkg['render'], gt))
        target, mask, target_normal, normal_mask = targets[camera.image_name]
        photo_weight = .5 if args.confidence_photo_weighting and float(mask.float().mean()) < .05 else 1.
        mask = mask & (pkg['rend_alpha'][0].detach() > .5)
        relative = (pkg['surf_depth'][0] - target) / target
        # Smooth absolute relative error limits outlier gradient, without clipping it.
        depth_loss = (torch.sqrt(relative[mask].square() + 1e-6) - .001).mean() if mask.any() else photo * 0
        normal = (1 - (pkg['rend_normal'] * pkg['surf_normal']).sum(dim=0)).mean()
        normal_mask = normal_mask & mask
        predicted_normal = torch.nn.functional.normalize(pkg['rend_normal'], dim=0)
        cosine = (predicted_normal * target_normal).sum(dim=0).abs().clamp(0, 1)
        normal_prior = (1 - cosine[normal_mask]).mean() if normal_mask.any() else photo * 0
        ramp = 1. if args.initial_ply else min(1., step / 500)
        normal_weight = opt.lambda_normal if args.initial_ply or step > 1000 else 0
        loss = photo_weight * photo + ramp * args.depth_weight * depth_loss + normal_weight * normal + ramp * args.normal_prior_weight * normal_prior
        if diffuse_mask is not None and diffuse_mask.any():
            loss = loss + args.structure_sh_penalty * model._features_rest[diffuse_mask].square().mean()
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss at {step}')
        loss.backward()
        if not args.initial_ply and step < 5000:
            with torch.no_grad():
                visible = pkg['visibility_filter']
                model.max_radii2D[visible] = torch.maximum(model.max_radii2D[visible], pkg['radii'][visible])
                model.add_densification_stats(pkg['viewspace_points'], visible)
                if step > 500 and step % 100 == 0 and len(model.get_xyz) < 1_000_000:
                    model.densify_and_prune(.0002, .05, scene.cameras_extent, 20 if step > 3000 else None)
                if step % 3000 == 0:
                    model.reset_opacity()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 250 == 0 or step == opt.iterations:
            row = dict(step=step, photo=float(photo.detach()), depth=float(depth_loss.detach()),
                       splats=len(model.get_xyz),
                       normal=float(normal.detach()), normal_prior=float(normal_prior.detach()), seconds=round(time.monotonic()-started, 1))
            history.append(row); print(json.dumps(row), flush=True)
        if step == opt.iterations:
            scene.save(step)
    (output / 'training-history.json').write_text(json.dumps(history, indent=2))
    print('Refinement finished', flush=True)


if __name__ == '__main__':
    main()
