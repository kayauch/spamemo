import numpy as np
from spatial_memory.regional_textures import regional_face_texture

def test_local_quality_replaces_poor_primary_without_blending():
    def project(i):
        return np.ones(64,bool),np.tile(([10]*4+[1]*4) if i==0 else ([1]*4+[10]*4),8)
    def sample(i,take):return np.tile([240,0,0] if i==0 else [0,0,240],(take.sum(),1))
    image,valid,source,*_=regional_face_texture((8,8),[0,0,0],2,project,sample)
    assert valid.all()
    assert (source.reshape(8,8)[:,0]==0).all()
    assert (source.reshape(8,8)[:,-1]==1).all()
    assert all(tuple(p) in ((240,0,0),(0,0,240)) for p in image)

def test_no_frame_cap_and_invalid_pixels_stay_empty():
    def project(i):
        mask=np.zeros(16,bool);mask[i]=True
        return mask,np.ones(16)
    def sample(i,take):return np.full((take.sum(),3),i+10)
    image,valid,source,_,_,selected,_=regional_face_texture((4,4),[2,2,2],12,project,sample)
    assert len(selected)==12
    assert valid.sum()==12
    assert (source[12:]==-1).all()
    assert (image[12:]==2).all()

def test_nonfinite_quality_cannot_claim_support():
    def project(i):return np.ones(4,bool),np.array([np.nan,np.inf,-1,0])
    def sample(i,take):raise AssertionError('must not sample invalid views')
    image,valid,*_=regional_face_texture((2,2),[1,2,3],1,project,sample)
    assert not valid.any()
