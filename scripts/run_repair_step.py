"""Windows/WSL bridge for isolated repair experiments. Never registers a model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_scene_appearance import ROOT, Runner

parser=argparse.ArgumentParser()
parser.add_argument('step',choices=['snapshot','train','control','evaluate'])
parser.add_argument('--scene',type=Path,required=True)
parser.add_argument('--colmap',type=Path,required=True)
parser.add_argument('--best',type=Path,required=True)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--distro',default='Ubuntu')
args=parser.parse_args()
scene,colmap,best,run=[getattr(args,k).resolve() for k in ('scene','colmap','best','run')]
if run.is_relative_to(scene) or run.is_relative_to(best) or run.is_relative_to(colmap):
    raise ValueError('Experiment must be outside original assets')
runner=Runner(args.distro)
common=['--eval','--data_device','cpu','-r',1,'--sh_degree',2]
run.mkdir(parents=True,exist_ok=True)
lock=ROOT/'data/appearance-pipeline.lock'
with lock.open('x') as f: f.write(str(run))
try:
    if args.step=='snapshot':
        commands=[('snapshot',runner.command(ROOT/'scripts/snapshot_scene_repair.py','-s',colmap,'-m',best,
            '--geometry-npz',scene/'reconstruction/reconstruction.npz','--output',run/'snapshot',*common))]
    elif args.step in ('train','control'):
        model_name='candidate' if args.step=='train' else 'control'
        extra=[] if args.step=='train' else ['--control-teacher',run/'snapshot']
        commands=[(args.step,runner.command(ROOT/'scripts/train_scene_repair.py','-s',run/'repair-colmap','-m',run/model_name,
            '--initial-ply',best/'point_cloud/iteration_3000/point_cloud.ply','--iterations',1500,*common,*extra))]
    else:
        # Evaluate both against the SAME unmodified real photographs/cameras.
        commands=[(name,runner.command(ROOT/'scripts/evaluate_2dgs.py','-s',colmap,'-m',model,
                    '--iteration',iteration,'--output-dir',run/name,'--repair-reference',run/'snapshot',
                    '--repair-spec',run/'repair-colmap',*common))
                  for name,model,iteration in [('best-evaluation',best,3000),('candidate-evaluation',run/'candidate',1500),('control-evaluation',run/'control',1500)]]
    for name,command in commands:
        log=run/f'{name}.log'
        if log.exists(): raise FileExistsError(log)
        print(name,flush=True)
        with log.open('w',encoding='utf-8') as file:
            subprocess.run(command,check=True,stdout=file,stderr=subprocess.STDOUT)
finally:
    lock.unlink()
