"""Repeat observed material patches in unobserved architecture; preserve evidence."""
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import trimesh

def select_patch(image, support):
    if not support.any(): return None
    median=np.median(image[support],axis=0)
    for size in (96,64,40,24,12):
        if min(support.shape)<size: continue
        valid=cv2.boxFilter(support.astype(np.float32),-1,(size,size),borderType=cv2.BORDER_CONSTANT)>.9999
        ys,xs=np.where(valid); best=None
        for y,x in zip(ys[::max(1,len(ys)//500)],xs[::max(1,len(xs)//500)]):
            y0=int(y-size//2);x0=int(x-size//2)
            tile=image[y0:y0+size,x0:x0+size]
            if tile.shape[:2]!=(size,size) or not support[y0:y0+size,x0:x0+size].all():continue
            low=cv2.GaussianBlur(tile.astype(np.float32),(0,0),3)
            score=float(np.linalg.norm(np.median(tile.reshape(-1,3),axis=0)-median)+2*np.std(low,axis=(0,1)).mean())
            if best is None or score<best[0]:best=(score,tile.copy(),x0,y0)
        if best is not None:return best
    return None

def tile_missing(image,keep,patch):
    _,tile,x,y=patch
    strip=np.concatenate([tile,tile[:,::-1]],axis=1)
    repeat=np.concatenate([strip,strip[::-1]],axis=0)
    yy,xx=np.indices(keep.shape);result=image.copy()
    pattern=repeat[(yy-y)%repeat.shape[0],(xx-x)%repeat.shape[1]]
    result[~keep]=pattern[~keep]
    return result

def complete_materials(directory,report):
    directory=Path(directory);assets={};patches={};kind_sources={}
    for s in report['surfaces']:
        if s['kind'] not in ('wall','ceiling'):continue
        image=np.array(Image.open(directory/s['texture']).convert('RGB'))
        keep=np.array(Image.open(directory/s['support']))>0
        extension=s.get('photo_extension',{}).get('mask')
        if extension:keep|=np.array(Image.open(directory/extension))>0
        assets[s['id']]=(image,keep)
        patch=select_patch(image,keep)
        if patch is not None:
            patches[s['id']]=patch;kind_sources.setdefault(s['kind'],[]).append(s['id'])
    scene=None;changed=[];unavailable=[]
    for s in report['surfaces']:
        if s['id'] not in assets:continue
        candidates=kind_sources.get(s['kind'],[])
        if not candidates:unavailable.append(s['id']);continue
        source=s['id'] if s['id'] in patches else min(candidates,key=lambda k:patches[k][0])
        image,keep=assets[s['id']];patch=patches[source]
        result=tile_missing(image,keep,patch)
        if scene is None:scene=trimesh.load(directory/'room.glb',force='scene',process=False)
        s['texture']=s['id']+'-material.png';Image.fromarray(result).save(directory/s['texture'])
        mask=s['id']+'-material-fill.png';Image.fromarray((~keep).astype(np.uint8)*255).save(directory/mask)
        s['pattern_completion']={'method':'mirrored_source_material_patch','source_surface':source,'source_rect':[patch[2],patch[3],patch[1].shape[1],patch[1].shape[0]],'mask':mask,'synthetic':True,'filled_fraction':float((~keep).mean())}
        scene.geometry[s['id']].visual.material.baseColorTexture=Image.fromarray(result);changed.append(s['id'])
    if scene is not None:(directory/'room.glb').write_bytes(scene.export(file_type='glb'))
    report['pattern_completion']={'enabled':bool(changed),'kinds':['wall','ceiling'],'synthetic':True,'strict_support_unchanged':True,'completed_surfaces':changed,'unavailable_surfaces':unavailable}
    report.setdefault('caveats',[]).append('Wall and ceiling holes may repeat a photographed material patch. This is synthetic appearance, not observed structure; support masks remain unchanged.')
    quality_path=directory/'quality.json'
    if quality_path.exists():
        quality=json.loads(quality_path.read_text(encoding='utf8'))
        quality['pattern_completion']=report['pattern_completion']
        quality_path.write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf8')
    (directory/'room.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report
