"""Run official SpatialLM in an isolated environment; preserve raw predictions."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/SpatialLM'))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from spatiallm import Layout
from spatiallm.model.sonata_encoder import Sonata
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors
from inference import preprocess_point_cloud
from spatial_memory.generation_evidence import box_evidence, transform_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--prediction', default='prediction-001')
    parser.add_argument('--model',choices=['SpatialLM1.1-Qwen-0.5B','SpatialLM1.1-Llama-1B'],default='SpatialLM1.1-Qwen-0.5B')
    parser.add_argument('--category',nargs='*',default=[])
    args = parser.parse_args()
    run = args.run.resolve()
    if not args.prediction.replace('-', '').isalnum():
        raise ValueError('Invalid prediction name')
    output = run / args.prediction
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    # Official Sonata supports standard attention, with the same checkpoint weights.
    # Avoid a separate FlashAttention build on Blackwell; record this deviation.
    original_init = Sonata.__init__
    def standard_attention(self, *a, **kw):
        kw['enable_flash'] = False
        original_init(self, *a, **kw)
    Sonata.__init__ = standard_attention
    checkpoint = ROOT / 'checkpoints' / args.model
    set_seed(args.seed)
    started = time.perf_counter()
    print('Loading official checkpoint', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, local_files_only=True)
    model.to('cuda')
    model.set_point_backbone_dtype(torch.float32)
    model.eval()
    cloud = load_o3d_pcd(str(run / 'spatiallm-input.ply'))
    points, colors = get_points_and_colors(cloud)
    bins = model.config.point_config['num_bins']
    tensor = preprocess_point_cloud(points, colors, Layout.get_grid_size(bins), bins)
    template = (ROOT / 'third_party/SpatialLM/code_template.txt').read_text()
    task = 'Detect walls, doors, windows, boxes.' if not args.category else 'Detect '+', '.join(args.category)+'.'
    prompt = '<|point_start|><|point_pad|><|point_end|>'+task+' The reference code is as followed: ' + template
    conversation=[dict(role='user',content=prompt)]
    if model.config.model_type=='spatiallm_qwen':conversation.insert(0,dict(role='system',content='You are a helpful assistant.'))
    ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True,
        return_tensors='pt').to(model.device)
    print(f'Inference: {len(points)} points; {ids.shape[1]} prompt tokens', flush=True)
    with torch.inference_mode():
        result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), point_clouds=tensor,
            max_new_tokens=args.max_tokens, do_sample=True, temperature=.6,
            top_p=.95, top_k=10, use_cache=True,pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(result[0, ids.shape[1]:], skip_special_tokens=True)
    (output / 'raw.txt').write_text(raw)
    # Layout is the official numeric parser; generated text is never executed.
    layout = Layout(raw)
    entities = layout.get_entities()
    candidate_lines = [l for l in raw.splitlines() if '=' in l and '(' in l]
    parse_complete = len(candidate_lines) == len(entities) and bool(entities)
    layout.undiscretize_and_unnormalize(bins)
    layout.translate(points.min(axis=0))
    (output / 'layout.txt').write_text(layout.to_language_string())
    adapter = json.loads((run / 'transform.json').read_text())
    inverse = np.asarray(adapter['generation_world_to_vg_world'])
    inverse_scale = np.linalg.norm(inverse[:3, 0])
    inverse_rotation = inverse[:3, :3] / inverse_scale
    boxes = []
    for box in layout.to_boxes():
        box = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in box.items()}
        if box['class'] == 'wall':
            wall = next(w for w in layout.walls if w.id == box['id'])
            box['scale'][1] = wall.thickness
        box['evidence'] = box_evidence(box['center'], box['rotation'], box['scale'], points,
            allow_planar=box['class'] in ('wall','door','window'))
        box['vg_world'] = dict(center=transform_points(np.array([box['center']]), inverse)[0].tolist(),
            rotation=(inverse_rotation @ np.asarray(box['rotation'])).tolist(),
            scale=(np.asarray(box['scale']) * inverse_scale).tolist())
        box['constraint'] = 'PROPOSED; requires evidence review'
        boxes.append(box)
    summary = dict(model='manycore-research/'+args.model,task=task,
        model_revision=json.loads((checkpoint/'source_revision.json').read_text()),
        upstream_revision='8913c44d84a450c53e9340b13317f8cf7144a738',
        layer='INFERRED_LAYOUT', accepted_for_generation=False,
        reason='Review walls, openings and object identity against observed frames first',
        seed=args.seed, attention='official Sonata enable_flash=False',
        input_sha256=hashlib.sha256((run/'spatiallm-input.ply').read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
        output_tokens=int(result.shape[1]-ids.shape[1]),
        token_limit_reached=bool(result.shape[1]-ids.shape[1]>=args.max_tokens),
        parse_complete=parse_complete, candidate_lines=len(candidate_lines),
        counts={k:len(getattr(layout,k)) for k in ('walls','doors','windows','bboxes')},
        metric_scale_verified=False, assumed_height_m=adapter['assumed_height_m'],
        entities=[dataclasses.asdict(e) for e in entities], boxes=boxes)
    (output/'layout.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('entities','boxes')},indent=2))


if __name__ == '__main__':
    main()
