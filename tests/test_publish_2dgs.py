import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_publish_preserves_baseline_and_rejects_mismatched_evaluation(tmp_path):
    scene = tmp_path / 'room'
    (scene / 'reconstruction').mkdir(parents=True)
    (scene / 'spatial_model').mkdir()
    poses = np.column_stack([np.eye(3), np.zeros(3)])[None]
    k = np.array([[[4.,0,2],[0,4,1],[0,0,1]]])
    npz = scene / 'reconstruction/reconstruction.npz'
    np.savez(npz, extrinsics=poses, intrinsics=k, depth=np.ones((1,2,4)))
    digest = hashlib.sha256(npz.read_bytes()).hexdigest()
    manifest = scene / 'spatial_model/manifest.json'
    manifest.write_text(json.dumps({'source_unchanged': True,'metrics':{'scene_diagonal':2}}))
    colmap = tmp_path / 'colmap'; colmap.mkdir()
    identity = np.eye(4).tolist()
    (colmap/'alignment.json').write_text(json.dumps({'scene_id':'room','source_npz_sha256':digest,'world_from_2dgs':identity}))
    model = tmp_path / 'model'
    (model/'evaluation').mkdir(parents=True)
    (model/'point_cloud/iteration_1').mkdir(parents=True)
    ply = model/'point_cloud/iteration_1/point_cloud.ply'
    evaluation = model/'evaluation/report.json'
    script = Path(__file__).resolve().parents[1]/'scripts/publish_2dgs_local.py'
    command = [sys.executable,str(script),str(scene),str(colmap),str(model)]
    hashes = []
    for body in (b'first evaluated model', b'second evaluated model'):
        ply.write_bytes(body)
        hashes.append(hashlib.sha256(body).hexdigest())
        evaluation.write_text(json.dumps({'iteration':1,'finite_parameters':True,'max_camera_center_error':0,
                                         'world_from_2dgs':identity,'model_ply_sha256':hashes[-1],'geometry_npz_sha256':digest}))
        subprocess.run(command,check=True,capture_output=True,text=True)
    saved = json.loads(manifest.read_text())
    assert saved['source_unchanged']
    assert saved['appearance_2dgs_baseline']['revision'] == hashes[0]
    assert saved['appearance_2dgs']['revision'] == hashes[1]
    assert saved['appearance_2dgs']['capture_views'][0]['image_aspect'] == 2
    before = manifest.read_bytes()
    ply.write_bytes(b'changed without re-evaluation')
    result = subprocess.run(command,capture_output=True,text=True)
    assert result.returncode != 0
    assert 'Evaluation report does not match' in result.stderr
    assert manifest.read_bytes() == before
    ply.write_bytes(b'second evaluated model')
    alignment=json.loads((colmap/'alignment.json').read_text())
    bad_poses=poses.copy(); bad_poses[0,0,3]=1
    alignment['appearance_camera_refinement']={'world_geometry_fixed':True,'intrinsics':k.tolist(),
                                             'extrinsics':bad_poses.tolist(),'source_npz_sha256':digest}
    (colmap/'alignment.json').write_text(json.dumps(alignment))
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode != 0
    assert 'exceeds bounded correction' in result.stderr
    assert manifest.read_bytes() == before
