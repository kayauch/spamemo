import numpy as np
from spatial_memory.photo_extension import extend_same_photos,horizontal_photo_regions

def test_extension_uses_selected_photo_pixels_without_changing_strict_evidence():
    texture=np.full((6,3),100,np.uint8);support=np.array([1,1,0,0,0,0],bool)
    original=texture.copy();mask=support.copy();called=[]
    def project(i):
        called.append(i)
        return (np.array([1,1,1,1,0,0],bool) if i==7 else np.array([0,0,1,1,1,0],bool)),np.ones(6)
    def sample(i,take):return np.tile([i,21,45],(take.sum(),1))
    result,filled,frames=extend_same_photos(texture,support,[7,9],project,sample)
    assert called==[7,9] and np.array_equal(texture,original) and np.array_equal(support,mask)
    assert result[:2].tolist()==[[100]*3]*2
    assert result[2:5].tolist()==[[7,21,45],[7,21,45],[9,21,45]]
    assert frames.tolist()==[-1,-1,7,7,9,-1]
    assert not filled[5] and result[5].tolist()==[100]*3

def test_no_selected_photo_does_not_repeat_or_invent_texture():
    result,filled,frames=extend_same_photos(np.full((4,3),88,np.uint8),np.zeros(4,bool),[],lambda _:None,lambda *_:None)
    assert not filled.any() and (frames==-1).all() and (result==88).all()

def test_horizontal_regions_reject_vertical_walls_and_invalid_depth():
    depth=np.ones((1,12,12));e=np.array([np.column_stack([np.eye(3),np.zeros(3)])]);k=np.array([[[8,0,6],[0,8,6],[0,0,1]]])
    m=np.eye(4);m[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
    assert not horizontal_photo_regions(depth,e,k,m).any()
    horizontal=horizontal_photo_regions(depth,e,k,np.eye(4))
    assert horizontal[0,1:-1,1:-1].all() and not horizontal[0,0].any()
