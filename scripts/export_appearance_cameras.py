"""Make an isolated COLMAP appearance dataset with bounded resection poses."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation

parser=argparse.ArgumentParser()
parser.add_argument('source',type=Path)
parser.add_argument('camera_report',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
if args.output.exists():raise ValueError('Choose a new dataset directory')
alignment=json.loads((args.source/'alignment.json').read_text())
report=json.loads(args.camera_report.read_text())
if report['source_npz_sha256']!=alignment['source_npz_sha256'] or not report['world_geometry_fixed']:
    raise ValueError('Mismatched geometry')
poses=np.array(report['extrinsics']);intrinsics=np.array(report['intrinsics'])
for e in poses:
    np.testing.assert_allclose(e[:,:3]@e[:,:3].T,np.eye(3),atol=1e-5)
    if np.linalg.det(e[:,:3])<.999:raise ValueError('Invalid camera rotation')
shutil.copytree(args.source,args.output)
sparse=args.output/'sparse/0'
point_rows={int(p[0]):p[:8] for l in (sparse/'points3D.txt').read_text().splitlines() if l and not l.startswith('#') for p in [l.split()]}
tracks={pid:[] for pid in point_rows}
old=[l for l in (sparse/'images.txt').read_text().splitlines() if not l.startswith('#')]
new=[];w,h=alignment['image_size']
for offset in range(0,len(old),2):
    parts=old[offset].split();image_id=int(parts[0]);frame=int(Path(parts[9]).stem)
    e,k=poses[frame],intrinsics[frame]
    xyzw=Rotation.from_matrix(e[:,:3]).as_quat()
    wxyz=xyzw[[3,0,1,2]]
    if wxyz[0]<0:wxyz=-wxyz
    parts[1:8]=[str(v) for v in np.r_[wxyz,e[:,3]]]
    new.append(' '.join(parts))
    obs=old[offset+1].split();ids=[int(obs[j]) for j in range(2,len(obs),3)]
    if not ids:new.append('');continue
    xyz=np.array([[float(v) for v in point_rows[pid][1:4]] for pid in ids])
    camera=xyz@e[:,:3].T+e[:,3]
    uv=camera@k.T;uv=uv[:,:2]/np.maximum(camera[:,2:],1e-8)
    inside=(camera[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    record=[]
    for pid,pixel in zip(np.array(ids)[inside],uv[inside]):
        tracks[pid].extend([str(image_id),str(len(record)//3)])
        record.extend([str(pixel[0]),str(pixel[1]),str(pid)])
    new.append(' '.join(record))
(sparse/'images.txt').write_text('\n'.join(new)+'\n')
(sparse/'points3D.txt').write_text('\n'.join(' '.join(row+tracks[pid]) for pid,row in point_rows.items())+'\n')
alignment['cameras_fixed']=False
alignment['appearance_camera_refinement']=report
alignment['camera_refinement_scope']='Appearance-only bounded poses; original VGGT point coordinates and intrinsics retained'
(args.output/'alignment.json').write_text(json.dumps(alignment,indent=2))
print(f"Exported {report['accepted_cameras']} bounded camera corrections; world points unchanged")
