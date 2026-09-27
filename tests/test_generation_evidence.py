import numpy as np
import pytest
from spatial_memory.generation_evidence import generation_transform, transform_points, box_evidence, vertical_anchors, anchor_wall_heights


def fixture():
    xyz=np.random.default_rng(7).uniform([-2,-1,0],[2,2,4],(1000,3))
    frame=dict(snap_enabled=True,room_to_world=np.eye(4).tolist(),confidence=.99)
    poses=np.tile(np.column_stack((np.eye(3),np.zeros(3))),(3,1,1))
    return xyz,frame,poses


def test_adapter_corrects_up_without_mirroring_and_roundtrips():
    xyz,frame,poses=fixture()
    out=generation_transform(xyz,frame,poses)
    m=np.asarray(out['vg_world_to_generation_world'])
    assert out['up_sign_flipped_from_existing_frame']
    assert np.linalg.det(m[:3,:3])>0
    assert (m[:3,:3]@np.array([0,-1,0]))[2]>0
    assert np.allclose(transform_points(transform_points(xyz,m),out['generation_world_to_vg_world']),xyz)
    assert frame['room_to_world']==np.eye(4).tolist()


def test_scale_is_explicit_hypothesis():
    xyz,frame,poses=fixture()
    out=generation_transform(xyz,frame,poses,3.)
    aligned=transform_points(xyz,out['vg_world_to_generation_world'])
    assert np.isclose(np.diff(np.quantile(aligned[:,2],[.005,.995]))[0],3.)
    assert out['metric_scale_verified'] is False


@pytest.mark.parametrize('direction,up',[('image-left',[-1,0,0]),('image-right',[1,0,0]),('image-bottom',[0,1,0])])
def test_sideways_video_preserves_handedness_and_records_orientation(direction,up):
    xyz,frame,poses=fixture()
    out=generation_transform(xyz,frame,poses,camera_up_axis=direction)
    m=np.asarray(out['vg_world_to_generation_world'])
    assert (m[:3,:3]@up)[2]>0
    assert np.linalg.det(m[:3,:3])>0
    assert out['camera_up_axis']==direction
    assert np.allclose(transform_points(transform_points(xyz,m),out['generation_world_to_vg_world']),xyz)


@pytest.mark.parametrize('fault',['disabled','rotation','camera','height','flat','nan'])
def test_invalid_evidence_cannot_silently_become_generation_input(fault):
    xyz,frame,poses=fixture();height=2.5
    if fault=='disabled':frame['snap_enabled']=False
    if fault=='rotation':frame['room_to_world'][0][0]=2
    if fault=='camera':poses[:]=np.nan
    if fault=='height':height=0
    if fault=='flat':xyz[:,1]=0
    if fault=='nan':xyz[0,0]=np.nan
    with pytest.raises(ValueError):generation_transform(xyz,frame,poses,height)


def test_rotated_box_counts_occupancy_without_claiming_semantics():
    r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    evidence=box_evidence([0,0,0],r,[4,1,1],[[0,1.5,0],[1.5,0,0]])
    assert evidence['points_inside']==1
    assert evidence['confidence'] is None
    assert not box_evidence([0,0,0],r,[4,0,1],[])['valid']
    assert not box_evidence([0,0,0],np.zeros((3,3)),[1,1,1],[])['valid']


def test_missing_ceiling_requires_explicit_hypothesis():
    planes=[dict(id='floor-candidate',source='plane',point_count=500,kind='wall',metadata={'normal':[0,0,1]},vertices=[[0,0,.1],[1,0,.1],[0,1,.1]])]
    with pytest.raises(ValueError):vertical_anchors(planes,np.eye(4))
    result=vertical_anchors(planes,np.eye(4),assumed_height_m=2.5)
    assert result['ceiling_observed'] is False
    assert result['ceiling']['z']==2.6
    assert result['ceiling']['source_component'] is None


def test_zero_thickness_wall_is_a_valid_surface_not_furniture():
    points=[[0,.04,0],[0,.07,0]]
    wall=box_evidence([0,0,0],np.eye(3),[4,0,2],points,allow_planar=True)
    assert wall['valid'] and wall['surface_only'] and wall['points_inside']==1
    assert not box_evidence([0,0,0],np.eye(3),[4,0,2],points)['valid']


def test_vertical_anchors_ignore_inverted_semantic_labels():
    comps=[dict(id=str(i),source='plane',kind=kind,point_count=500,metadata={'normal':[0,0,1]},
        vertices=[[0,0,z],[1,0,z],[1,1,z],[0,1,z]]) for i,(kind,z) in enumerate([('ceiling',0),('floor',2.5)])]
    anchors=vertical_anchors(comps,np.eye(4))
    assert anchors['floor']['source_component']=='0'
    assert anchors['ceiling']['source_component']=='1'
    with pytest.raises(ValueError):vertical_anchors(comps[:1],np.eye(4))


def test_wall_correction_preserves_furniture_openings_and_source():
    boxes=[dict(id=i,**{'class':kind},center=[1,1,1.4],scale=[2,0,2.8],rotation=np.eye(3).tolist())
        for i,kind in enumerate(['wall','door','bbox'])]
    result,changes=anchor_wall_heights(boxes,{'floor':{'z':0},'ceiling':{'z':2.5}})
    assert len(changes)==1 and result[0]['scale'][2]==2.5
    assert result[1:]==boxes[1:] and boxes[0]['scale'][2]==2.8
    result,changes=anchor_wall_heights(boxes,{'floor':{'z':0},'ceiling':{'z':4.5}})
    assert result==boxes and not changes
