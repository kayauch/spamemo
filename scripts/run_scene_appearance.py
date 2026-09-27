"""Shared VGGT -> COLMAP -> 2DGS -> validation workflow, on Windows + WSL.

Default is read-only preflight. --run trains into a fresh directory.
--publish additionally registers an accepted candidate, never a failed candidate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from spatial_memory.quality_gate import POLICY_VERSION, compare_reports


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def preflight(scene):
    import numpy as np
    metrics = read(scene / 'reconstruction/metrics.json')
    if metrics.get('model') != 'VGGT-1B' or metrics.get('fallback'):
        raise ValueError('A real VGGT reconstruction is required')
    frames = read(scene / 'frames/manifest.json')['frames']
    if len(frames) < 17:
        raise ValueError('Need at least 17 frames for three held-out views; capture a longer video')
    for frame in frames:
        path = (scene / 'frames' / frame['name']).resolve()
        if not path.is_relative_to(scene / 'frames') or not path.is_file():
            raise ValueError('Missing or invalid source frame')
    with np.load(scene / 'reconstruction/reconstruction.npz', allow_pickle=False) as data:
        depth = data['depth']
        if depth.ndim == 4: depth = depth[..., 0]
        n, h, w = depth.shape
        if n != len(frames) or data['extrinsics'].shape != (n, 3, 4) or data['intrinsics'].shape != (n, 3, 3):
            raise ValueError('Frame and camera counts differ')
        if data['confidence'].shape[:3] != depth.shape:
            raise ValueError('Confidence and depth dimensions differ')
        if not np.allclose(data['intrinsics'][:, :2, 2], [w/2, h/2], atol=1e-4):
            raise ValueError('Principal point must be centered for the official 2DGS renderer')
    for name in ('reconstruction/scene.ply', 'spatial_model/manifest.json'):
        if not (scene / name).is_file(): raise ValueError(f'Missing {name}')
    return dict(scene=str(scene), frames=n, held_out_views=len(range(0, n, 8)),
                image_size=[w, h], policy=POLICY_VERSION)


class Runner:
    def __init__(self, distro):
        self.distro = distro

    def path(self, path):
        if os.name != 'nt': return str(path)
        return subprocess.check_output(['wsl.exe', '-d', self.distro, '--exec', 'wslpath', '-a', path.as_posix()], text=True).strip()

    def command(self, script, *args):
        values = [self.path(v) if isinstance(v, Path) else str(v) for v in args]
        command = [self.path(ROOT / 'third_party/2dgs-env/bin/python'), self.path(script), *values]
        if os.name == 'nt':
            command = ['wsl.exe', '-d', self.distro, '--cd', self.path(ROOT / 'third_party/2dgs'), '--exec',
                       'env', 'OMP_NUM_THREADS=4', 'MPLBACKEND=Agg', 'PYTHONUNBUFFERED=1', *command]
        return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('--output', type=Path, help='Fresh run directory; required with --run')
    parser.add_argument('--distro', default='Ubuntu')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    scene = args.scene.resolve()
    status = preflight(scene)
    print(json.dumps(status, indent=2), flush=True)
    if args.publish and not args.run: parser.error('--publish requires --run')
    if not args.run: return
    if args.output is None: parser.error('--run requires --output')
    output = args.output.resolve()
    if output.is_relative_to(scene): parser.error('Output must be outside the source scene')
    runner = Runner(args.distro)
    # Check GPU/environment before creating a run or exporting frames.
    python = ROOT / 'third_party/2dgs-env/bin/python'
    probe = [runner.path(python), '-c', 'import torch, diff_surfel_rasterization; assert torch.cuda.is_available(), "CUDA unavailable"']
    if os.name == 'nt': probe = ['wsl.exe', '-d', args.distro, '--exec', *probe]
    subprocess.run(probe, check=True)
    output.mkdir(parents=True, exist_ok=False)
    lock = ROOT / 'data/appearance-pipeline.lock'
    lock.parent.mkdir(exist_ok=True)
    # O_EXCL prevents two pipeline jobs from training on the same GPU.
    with lock.open('x', encoding='utf-8') as file:
        file.write(json.dumps(dict(pid=os.getpid(), output=str(output))))
    status.update(state='running', started=datetime.now(timezone.utc).isoformat(),
                  settings=dict(baseline_steps=7000, refinement_steps=3000, depth_weight=.1,
                                normal_prior_weight=0, max_points=50000, cameras_fixed=True,
                                export_seed=42, baseline_seed=0, refinement_seed=42),
                  code_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [Path(__file__), ROOT/'src/spatial_memory/quality_gate.py',
                                            ROOT/'src/spatial_memory/colmap_export.py', ROOT/'scripts/refine_2dgs.py',
                                            ROOT/'scripts/evaluate_2dgs.py', ROOT/'third_party/2dgs/train.py']})

    def step(name, command):
        status.update(step=name)
        save(output / 'status.json', status)
        print(name, flush=True)
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           cwd=ROOT / 'third_party/2dgs', env={**os.environ, 'OMP_NUM_THREADS':'4', 'MPLBACKEND':'Agg',
                                                             'PYTHONPATH':str(ROOT/'src')})

    try:
        from spatial_memory.colmap_export import export_scene
        colmap = output / 'colmap'
        status['step'] = 'export'
        save(output / 'status.json', status)
        export_scene(scene, colmap, 50000)
        baseline, candidate = output / 'baseline', output / 'candidate'
        geometry = scene / 'reconstruction/reconstruction.npz'
        common = ['--sh_degree', 2, '--eval', '--data_device', 'cpu', '-r', 1]
        step('train-baseline', runner.command(ROOT/'third_party/2dgs/train.py', '-s', colmap, '-m', baseline,
             '--iterations', 7000, '--position_lr_max_steps', 7000, '--densify_until_iter', 5000,
             '--save_iterations', 7000, '--test_iterations', 7000, '--port', 6019, *common))
        step('evaluate-baseline', runner.command(ROOT/'scripts/evaluate_2dgs.py', '-s', colmap, '-m', baseline,
             '--iteration', 7000, '--geometry-npz', geometry, *common))
        step('refine', runner.command(ROOT/'scripts/refine_2dgs.py', '-s', colmap, '-m', candidate,
             '--iterations', 3000, '--geometry-npz', geometry, '--depth-weight', .1,
             '--initial-ply', baseline/'point_cloud/iteration_7000/point_cloud.ply', *common))
        step('evaluate-candidate', runner.command(ROOT/'scripts/evaluate_2dgs.py', '-s', colmap, '-m', candidate,
             '--iteration', 3000, '--geometry-npz', geometry, *common))
        decision = compare_reports(read(baseline/'evaluation/report.json'), read(candidate/'evaluation/report.json'))
        save(output / 'decision.json', decision)
        status.update(decision=decision, selected_model=str(candidate if decision['accepted'] else baseline), published=False)
        if args.publish and decision['accepted']:
            step('publish', [sys.executable, str(ROOT/'scripts/publish_2dgs_local.py'), str(scene), str(colmap), str(candidate)])
            status['published'] = True
            step('fusion', [sys.executable, str(ROOT/'scripts/build_fusion_shell.py'), str(scene/'spatial_model/manifest.json')])
            status['fusion_built'] = True
        status['state'] = 'complete'
    except BaseException as error:
        status.update(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        save(output / 'status.json', status)
        lock.unlink()
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
