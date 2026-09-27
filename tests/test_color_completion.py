import numpy as np
from spatial_memory.color_completion import fill_missing_color

def test_only_missing_pixels_are_filled_and_original_evidence_stays_intact():
    image=np.zeros((32,32,3),np.uint8);image[:,:8]=[145,142,136]
    support=np.zeros((32,32),bool);support[:,:8]=True
    original=support.copy();region=np.ones_like(support);region[-1]=False
    result,filled=fill_missing_color(image,support,region)
    assert np.array_equal(result[support],image[support])
    assert np.array_equal(support,original)
    assert np.all(result[filled]==[145,142,136])
    assert not filled[support].any() and not filled[-1].any()

def test_no_evidence_keeps_original_unless_same_material_palette_exists():
    image=np.full((10,10,3),80,np.uint8);support=np.zeros((10,10),bool)
    unchanged,fill=fill_missing_color(image,support,~support)
    assert np.array_equal(unchanged,image) and not fill.any()
    result,fill=fill_missing_color(image,support,~support,[180,175,160])
    assert fill.all() and np.all(result==[180,175,160])
