import numpy as np
from spatial_memory.layout_review import lift_edge,top_proposal,supported_edge_height


def box():return dict(center=[0,0,1],scale=[4,1,2],rotation=np.eye(3).tolist())
def points(z):return np.column_stack((np.linspace(-1,1,40),np.zeros(40),np.full(40,z)))


def test_backprojection_and_confidence_filter():
    depth=np.ones((100,100))*2;conf=np.ones_like(depth)*2
    k=np.array([[100,0,50],[0,100,50],[0,0,1.]])
    e=np.column_stack((np.eye(3),[0,0,1]));m=np.eye(4);m[2,3]=3
    xyz=lift_edge([[10,50],[90,50]],depth,conf,k,e,m)
    assert np.allclose(xyz[:,1],0) and np.allclose(xyz[:,2],4)
    assert len(lift_edge([[10,50],[90,50]],depth,conf*.5,k,e,m))==0
    assert len(lift_edge([[10,50],[11,50]],depth,conf,k,e,m))==2


def test_top_adjustment_preserves_other_dimensions_and_input():
    original=box();result,report=top_proposal(original,[dict(frame=1,points=points(1.8)),dict(frame=2,points=points(1.82))])
    assert report['accepted'] and np.isclose(result['scale'][2],1.81)
    assert result['center'][:2]==original['center'][:2]
    assert result['scale'][:2]==original['scale'][:2] and original['scale'][2]==2
    assert np.isclose(result['center'][2]-result['scale'][2]/2,0)


def test_duplicate_views_and_disagreement_cannot_approve():
    original=box()
    for obs in ([dict(frame=1,points=points(1.8))]*2,[dict(frame=1,points=points(1.6)),dict(frame=2,points=points(1.9))]):
        result,report=top_proposal(original,obs)
        assert not report['accepted'] and result==original


def test_other_object_and_noisy_edge_rejected():
    p=points(1.8);p[:,0]+=10
    assert supported_edge_height(box(),p) is None
    p=points(1.8);p[:,2]=np.linspace(1,2,40)
    assert supported_edge_height(box(),p) is None
    assert supported_edge_height(box(),[]) is None
