import numpy as np
import pytest
from spatial_memory.solid_room import wall_loop, photo_visibility
from spatial_memory.solid_room import closed_shell, furniture_parts, navigation_spawn, furniture_occlusion
import trimesh


def test_concave_perimeter_keeps_recess_and_rejects_open_room():
    vertices=np.array([[0,0],[3,0],[3,1],[1,1],[1,3],[0,3]])
    boxes=[]
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        d=b-a;length=np.linalg.norm(d);x,y=d/length
        boxes.append(dict(**{'class':'wall'},center=[*((a+b)/2),1],scale=[length,0,2],rotation=[[x,-y,0],[y,x,0],[0,0,1]]))
    loop=wall_loop(boxes[::-1])
    area=abs(np.sum(loop[:,0]*np.roll(loop[:,1],-1)-loop[:,1]*np.roll(loop[:,0],-1)))/2
    assert area==5  # A hull would fill the recess.
    with pytest.raises(ValueError):wall_loop(boxes[:-1])


def test_photo_projection_rejects_foreground_and_bad_confidence():
    k=np.array([[10,0,5],[0,10,5],[0,0,1]])
    e=np.column_stack([np.eye(3),np.zeros(3)])
    depth=np.ones((12,12))*2;confidence=np.ones_like(depth)*2
    points=np.array([[0,0,2],[0,0,3],[0,0,-2],[5,0,2]])
    uv,valid,z=photo_visibility(points,e,k,depth,confidence,.1)
    assert valid.tolist()==[True,False,False,False]
    assert np.allclose(uv[0],[5,5])
    confidence[5,5]=1
    assert not photo_visibility(points,e,k,depth,confidence,.1)[1].any()


@pytest.mark.parametrize('reverse',[False,True])
def test_concave_shell_is_closed_with_correct_volume(reverse):
    points=np.array([[0,0],[3,0],[3,1],[1,1],[1,3],[0,3]])
    if reverse:points=points[::-1]
    vertices,faces=closed_shell(points,.2,2.7)
    mesh=trimesh.Trimesh(vertices,faces)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume==pytest.approx(12.5)


def test_inferred_desk_supports_reach_floor_without_changing_observed_top():
    box=dict(label='desk',center=[2,2,.75],scale=[1,.6,.06],rotation=np.eye(3))
    parts=furniture_parts(box,.1)
    assert len(parts)==5 and parts[0]['inferred_completion'] is False
    assert parts[0]['center'][2]+parts[0]['scale'][2]/2==pytest.approx(.78)
    for leg in parts[1:]:
        assert leg['center'][2]-leg['scale'][2]/2==pytest.approx(.1)
        assert leg['inferred_completion']


def test_model_cannot_be_declared_walkable_when_floor_is_blocked():
    polygon=[[0,0],[2,0],[2,2],[0,2]]
    box=dict(center=[1,1,.5],scale=[4,4,1],rotation=np.eye(3).tolist(),bottom=0,top=1)
    with pytest.raises(ValueError,match='free floor'):
        navigation_spawn(polygon,[box],0,1.5)
    assert navigation_spawn(polygon,[],0,1.5)[2]==1.5


def test_furniture_photos_are_not_baked_into_wall_behind_them():
    box=dict(center=[0,0,2],scale=[1,1,1],rotation=np.eye(3))
    points=np.array([[0,0,4],[3,0,4],[0,0,1],[0,0,-4]])
    assert furniture_occlusion(points,np.zeros(3),[box]).tolist()==[True,False,False,False]
