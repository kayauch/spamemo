import sys, tempfile, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from spatial_memory.material_completion import select_patch,tile_missing,complete_materials
from spatial_memory.semantic_textures import SemanticPhotoMasks

def test_patch_preserves_observed_pixels():
    y,x=np.indices((100,100));image=np.stack([x%8*20,y%8*20,x%8*20],-1).astype('uint8');mask=np.zeros((100,100),bool);mask[10:90,10:90]=True
    patch=select_patch(image,mask);assert patch is not None
    output=tile_missing(image,mask,patch);assert np.array_equal(output[mask],image[mask]);assert output[~mask].std()>10
    assert select_patch(image,np.zeros_like(mask)) is None

def test_wall_margin_keeps_furniture_blocked():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);wall=np.zeros((80,80),'uint8');wall[20:60,20:50]=255;blocked=np.zeros_like(wall);blocked[20:60,50:60]=255
        Image.fromarray(wall).save(p/'wall.png');Image.fromarray(blocked).save(p/'box.png')
        layout={'image_furniture':{'version':3,'detections':[{'frame_index':0,'kind':'wall','mask_path':'wall.png'},{'frame_index':0,'kind':'furniture','mask_path':'box.png'}]}}
        masks=SemanticPhotoMasks(p,layout,80,80);uv=np.array([[15,40],[54,40],[3,40]])
        assert not masks.allowed(0,uv,'wall').any()
        assert masks.allowed(0,uv,'wall',wall_margin=12).tolist()==[True,False,False]

