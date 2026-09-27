import numpy as np
from PIL import Image
from spatial_memory.wall_evidence import depth_normals, matching_normals, independent_support
from spatial_memory.semantic_textures import SemanticPhotoMasks

def test_duplicate_origins_do_not_count_as_multiple_views():
    def project(i):return np.array([True,i>1]),np.ones(2)
    allowed,votes=independent_support(['a','a','b','c'],2,project)
    assert votes.tolist()==[3,2]
    assert allowed.tolist()==[True,False]

def test_plane_normals_and_wrong_direction():
    depth=np.full((1,10,10),2.)
    normals=depth_normals(depth,np.eye(4)[None,:3],np.eye(3)[None])
    uv=np.array([[4.,4.],[np.nan,1],[-1,2]])
    assert matching_normals(normals[0],uv,[0,0,1]).tolist()==[True,False,False]
    assert not matching_normals(normals[0],uv,[1,0,0]).any()

def test_geometry_permission_keeps_furniture_and_unknown_exclusions(tmp_path):
    records=[]
    for identity,kind,x in [('f','furniture',4),('u','unknown',10)]:
        mask=np.zeros((20,20),np.uint8);mask[3:8,x:x+3]=255
        Image.fromarray(mask).save(tmp_path/f'{identity}.png')
        records.append(dict(detection_id=identity,kind=kind,frame_index=0,mask_path=f'{identity}.png'))
    masks=SemanticPhotoMasks(tmp_path,{'image_furniture':{'version':3,'detections':records}},20,20)
    uv=np.array([[5,5],[11,5],[17,17]])
    assert not masks.allowed(0,uv,'wall').any()
    assert masks.allowed(0,uv,'wall',allow_unlabelled_wall=True).tolist()==[False,False,True]
    (tmp_path/'f.png').unlink()
    masks.cache.clear()
    assert not masks.allowed(0,uv,'wall',allow_unlabelled_wall=True).any()
