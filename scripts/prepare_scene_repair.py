"""Plan a multi-view repair, then ingest an AI proposal without changing good pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
from spatial_memory.scene_repair import correspondence,sample,composite_locked,AI_REPAIRED


def read_image(path): return np.array(Image.open(path).convert('RGB'))
def geometry(snapshot,frame):
    with np.load(snapshot/frame/'geometry.npz',allow_pickle=False) as file: return dict(file)
def warp(target,source):
    return correspondence(target['depth'],target['extrinsic'],target['intrinsic'],source['depth'],source['extrinsic'],source['intrinsic'])


def plan(snapshot,output):
    metadata=json.loads((snapshot/'snapshot.json').read_text())
    train=[r for r in metadata['frames'] if not r['held_out']]
    cache={r['frame']:geometry(snapshot,r['frame']) for r in train}
    best=None
    for row in sorted(train,key=lambda r:r['repair_pixels'],reverse=True)[:20]:
        anchor=row['frame']; src=cache[anchor]
        source_mask=np.asarray(Image.open(snapshot/anchor/'mask.png'))>0
        if not source_mask.any(): continue
        neighbors=[]
        for target_row in train:
            target=target_row['frame']
            if target==anchor or abs(int(target)-int(anchor))>6: continue
            dst=cache[target]
            uv,valid=warp(src,dst)
            support=valid & sample((dst['broken'] & dst['support']).astype(np.uint8),uv,True).astype(bool)
            neighbors.append((target,support))
        if len(neighbors)<2: continue
        votes=sum(s.astype(np.uint8) for _,s in neighbors)
        allowed=source_mask & (votes>=2)
        if allowed.sum()<20: continue
        selected=[]
        for target,_ in neighbors:
            dst=cache[target]; uv,valid=warp(dst,src)
            mask=valid & dst['broken'] & dst['support'] & sample(allowed.astype(np.uint8),uv,True).astype(bool)
            if mask.sum()>=10: selected.append((target,int(mask.sum())))
        selected=sorted(selected,key=lambda x:x[1],reverse=True)[:2]
        if len(selected)<2: continue
        # Require the final chosen two views, not just any two in the search set.
        named=dict(neighbors)
        allowed &= named[selected[0][0]] & named[selected[1][0]]
        if allowed.sum()<20: continue
        score=int(allowed.sum())
        if best is None or score>best[0]: best=(score,anchor,selected,allowed)
    if best is None: raise ValueError('No repair region has two additional depth-consistent training views; do not generate')
    score,anchor,selected,allowed=best
    output.mkdir(parents=True,exist_ok=False)
    Image.fromarray(allowed.astype(np.uint8)*255).save(output/'mask.png')
    for name in ('render','observed'): shutil.copy2(snapshot/anchor/f'{name}.png',output/f'{name}.png')
    overlay=read_image(output/'render.png'); overlay[allowed]=[255,0,180]
    Image.fromarray(overlay).save(output/'region.png')
    info=dict(anchor=anchor,targets=[t for t,_ in selected],pixels=score,snapshot=str(snapshot.resolve()),
              source_npz_sha256=metadata['source_npz_sha256'],source_ply_sha256=metadata['source_ply_sha256'],
              policy='observed broken pixels only; two additional training views; 2.5% depth tolerance',
              generation_status='pending',unknown_generation=False)
    (output/'plan.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))


def apply(plan_dir,proposal,colmap,output):
    info=json.loads((plan_dir/'plan.json').read_text())
    snapshot=Path(info['snapshot'])
    alignment=colmap/'alignment.json'
    meta=json.loads((snapshot/'snapshot.json').read_text())
    if hashlib.sha256(alignment.read_bytes()).hexdigest()!=meta['alignment_sha256']: raise ValueError('Camera alignment changed')
    anchor=info['anchor']; src=geometry(snapshot,anchor)
    mask=np.asarray(Image.open(plan_dir/'mask.png'))>0
    original=read_image(plan_dir/'render.png')
    with Image.open(proposal) as image:
        size=image.size
        if abs(image.width/image.height-original.shape[1]/original.shape[0])>.04: raise ValueError('AI changed aspect ratio')
        # Normalize tool output resolution only; never spatially align invented geometry.
        proposed=np.asarray(image.convert('RGB').resize((original.shape[1],original.shape[0]),Image.Resampling.LANCZOS))
    anchor_repair=composite_locked(original,proposed,mask)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(colmap/'sparse',output/'sparse')
    shutil.copy2(alignment,output/'alignment.json')
    (output/'images').mkdir(); (output/'provenance').mkdir()
    changed=[]
    for row in meta['frames']:
        frame=row['frame']; target=geometry(snapshot,frame)
        image=read_image(snapshot/frame/'render.png')
        patch=np.zeros(image.shape[:2],bool)
        if row['held_out']:
            shutil.copy2(colmap/'images'/f'{frame}.png',output/'images'/f'{frame}.png')
            continue
        if frame==anchor: patch=mask; repaired=anchor_repair
        elif frame in info['targets']:
            uv,valid=warp(target,src)
            patch=valid & target['support'] & target['broken'] & sample(mask.astype(np.uint8),uv,True).astype(bool)
            projected=sample(anchor_repair,uv)
            repaired=composite_locked(image,projected,patch)
        else: repaired=image
        Image.fromarray(repaired).save(output/'images'/f'{frame}.png')
        labels=target['labels'].copy(); labels[patch]=AI_REPAIRED
        Image.fromarray(labels).save(output/'provenance'/f'{frame}.png')
        if patch.any():
            if not np.array_equal(image[~patch],repaired[~patch]): raise ValueError('Locked pixels changed')
            changed.append(dict(frame=frame,pixels=int(patch.sum())))
    consistency=dict(passed=len(changed)==3,changed_views=changed,
        method='one AI anchor, depth-checked reprojection into two observed broken regions; no independently generated views',
        depth_tolerance=.025,independent_validation=False,
        limitation='Construction enforces shared projected color, but cannot prove the AI patch is physically correct')
    if not consistency['passed']: raise ValueError('Insufficient supported repair views')
    info['generation_status']='proposal_ingested'
    report=dict(**info,proposal_sha256=hashlib.sha256(proposal.read_bytes()).hexdigest(),proposal_size=size,
                normalized_size=[original.shape[1],original.shape[0]],consistency=consistency,
                good_pixels_locked_in_targets=True,held_out_images_unchanged=True,
                train_target='frozen best render, with AI pixels only inside approved masks')
    (output/'repair.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['plan','apply'])
    p.add_argument('--snapshot',type=Path); p.add_argument('--plan',type=Path)
    p.add_argument('--proposal',type=Path); p.add_argument('--colmap',type=Path)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.mode=='plan': plan(a.snapshot,a.output)
    else: apply(a.plan,a.proposal,a.colmap,a.output)
