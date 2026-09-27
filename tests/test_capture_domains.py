import numpy as np
from PIL import Image

from spatial_memory.texturing import preprocess_frame
from spatial_memory.generation_evidence import generation_transform


def test_mixed_aspect_camera_domain_keeps_batch_padding(tmp_path):
    path=tmp_path/'landscape.jpg'
    Image.new('RGB',(1036,588),(80,100,120)).save(path)
    im=np.asarray(preprocess_frame(path,518,518))
    assert im.shape==(518,518,3)
    assert (im[:112]==255).all() and (im[406:]==255).all()
    assert np.allclose(im[112:406].mean((0,1)),[80,100,120],atol=2)
    enlarged=np.asarray(preprocess_frame(path,518,518,scale=2))
    assert (enlarged[:224]==255).all() and (enlarged[812:]==255).all()


def test_sensor_photo_uses_direct_camera_domain(tmp_path):
    path=tmp_path/'sensor.png'
    Image.new('RGB',(640,480),(10,30,90)).save(path)
    result=np.asarray(preprocess_frame(path,240,320,preprocessing='aligned_rgbd'))
    assert result.shape==(240,320,3) and (result[0,0]==[10,30,90]).all()


def test_metric_capture_does_not_get_rescaled_to_assumed_ceiling():
    xyz=np.random.default_rng(7).uniform([-2,-1,0],[2,2,4],(1000,3))
    frame=dict(snap_enabled=True,room_to_world=np.eye(4).tolist(),confidence=.99)
    poses=np.tile(np.column_stack((np.eye(3),np.zeros(3))),(3,1,1))
    transform=generation_transform(xyz,frame,poses,preserve_metric_scale=True)
    matrix=np.asarray(transform['vg_world_to_generation_world'])
    assert np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3))
    assert transform['metric_scale_calibrated'] is True
    assert transform['metric_scale_verified'] is False
