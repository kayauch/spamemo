"""Download official local image models. No room images are uploaded."""
import json
from pathlib import Path
from huggingface_hub import model_info, snapshot_download

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    for repo,name in [('IDEA-Research/grounding-dino-tiny','grounding-dino-tiny'),('facebook/sam2.1-hiera-tiny','sam2.1-hiera-tiny')]:
        path=ROOT/'checkpoints'/name;record=path/'source_revision.json'
        revision=json.loads(record.read_text())['revision'] if record.exists() else model_info(repo).sha
        snapshot_download(repo,revision=revision,local_dir=path,allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
        record.write_text(json.dumps(dict(repo=repo,revision=revision)),encoding='utf-8')
        print(name,'ready')
