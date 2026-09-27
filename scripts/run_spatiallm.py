"""Windows entry point: isolate SpatialLM and use the shared GPU lock."""
import argparse
from pathlib import Path
import subprocess
from run_scene_appearance import ROOT, Runner

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--distro',default='Ubuntu')
p.add_argument('--attempt',type=int,default=1)
p.add_argument('--model',choices=['SpatialLM1.1-Qwen-0.5B','SpatialLM1.1-Llama-1B'],default='SpatialLM1.1-Qwen-0.5B')
p.add_argument('--category',nargs='*',default=[])
a=p.parse_args();run=a.run.resolve();runner=Runner(a.distro)
if a.attempt<1:raise ValueError('Attempt must be positive')
if not (run/'transform.json').is_file():raise ValueError('Prepare evidence first')
command=['wsl.exe','-d',a.distro,'--exec','env','OMP_NUM_THREADS=4','PYTHONUNBUFFERED=1',
    'CUDA_HOME='+runner.path(ROOT/'third_party/cuda-12.8'),
    runner.path(ROOT/'third_party/spatiallm-env/bin/python'),
    runner.path(ROOT/'scripts/infer_spatiallm.py'),runner.path(run),'--prediction',f'prediction-{a.attempt:03d}',
    '--model',a.model]
if a.category:command+=['--category',*a.category]
lock=ROOT/'data/appearance-pipeline.lock'
with lock.open('x') as file:file.write(str(run))
try:
    with (run/f'inference-{a.attempt:03d}.log').open('x',encoding='utf-8') as log:
        print('SpatialLM running; log:',log.name,flush=True)
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
finally:
    lock.unlink()
