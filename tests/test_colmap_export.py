import json
import numpy as np
import open3d as o3d
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation
from spatial_memory.colmap_export import colmap_pose, export_scene


def test_pose_keeps_world_to_camera_convention():
    r = Rotation.from_euler('xyz', [.3, -.7, 1.2]).as_matrix()
    t = np.array([.1, -1.2, 2.])
    q, actual_t = colmap_pose(np.column_stack([r, t]))
    rebuilt = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    points = np.array([[1,2,3], [-1,.5,4]])
    np.testing.assert_allclose(points @ rebuilt.T + actual_t, points @ r.T + t)
    np.testing.assert_allclose(-rebuilt.T @ actual_t, -r.T @ t)


def test_reject_invalid_camera():
    with pytest.raises(ValueError):
        colmap_pose(np.zeros((3,4)))


def test_export_preserves_coordinates_and_observation_links(tmp_path):
    scene = tmp_path / 'scene'
    (scene/'reconstruction').mkdir(parents=True)
    (scene/'frames').mkdir()
    Image.new('RGB',(518,294),(128,64,32)).save(scene/'frames/0.png')
    (scene/'frames/manifest.json').write_text(json.dumps({'frames':[{'name':'0.png'}]}))
    (scene/'reconstruction/metrics.json').write_text(json.dumps({'model':'VGGT-1B'}))
    k=np.array([[400.,0,259],[0,400,147],[0,0,1]])
    e=np.column_stack([np.eye(3),np.zeros(3)])
    np.savez(scene/'reconstruction/reconstruction.npz',depth=np.full((1,294,518),2.),extrinsics=e[None],intrinsics=k[None])
    points=np.array([[0.,0,2],[.1,.1,2],[-.1,0,2]])
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors=o3d.utility.Vector3dVector(np.full_like(points,.5))
    o3d.io.write_point_cloud(str(scene/'reconstruction/scene.ply'),cloud)
    out=tmp_path/'colmap'
    result=export_scene(scene,out,3)
    np.testing.assert_array_equal(result['world_from_2dgs'],np.eye(4))
    lines=(out/'sparse/0/points3D.txt').read_text().splitlines()[1:]
    for i,line in enumerate(lines):
        values=line.split()
        np.testing.assert_allclose(np.array(values[1:4],float),points[i])
        assert values[8:]==['1',str(i)]
    observations=(out/'sparse/0/images.txt').read_text().splitlines()[-1].split()
    assert observations[2::3]==['1','2','3']
    assert Image.open(out/'images/000000.png').size==(518,294)
    with pytest.raises(FileExistsError):
        export_scene(scene,out)
    k[0,2]+=2
    np.savez(scene/'reconstruction/reconstruction.npz',depth=np.full((1,294,518),2.),extrinsics=e[None],intrinsics=k[None])
    with pytest.raises(ValueError,match='principal'):
        export_scene(scene,tmp_path/'offset')
