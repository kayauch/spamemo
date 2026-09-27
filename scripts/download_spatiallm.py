"""Download only the official 0.5B inference checkpoint into ignored storage."""
import json
import argparse
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--model',choices=['SpatialLM1.1-Qwen-0.5B','SpatialLM1.1-Llama-1B'],default='SpatialLM1.1-Qwen-0.5B')
a=p.parse_args()
repo='manycore-research/'+a.model
info=HfApi().model_info(repo)
folder=root/'checkpoints'/a.model
snapshot_download(repo_id=repo,revision=info.sha,local_dir=folder,
                  allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.md'])
(folder/'source_revision.json').write_text(json.dumps(dict(repo=repo,revision=info.sha),indent=2))
print(json.dumps(dict(path=str(folder),revision=info.sha)))
