"""Write the reproducible feedback log and a same-size before/after contact sheet."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw

parser=argparse.ArgumentParser()
parser.add_argument('--chosen',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
data=root/'data/2dgs-room'
baseline=json.loads((data/'train-7000/evaluation/report.json').read_text())
base_views={r['frame']:r for r in baseline['test_views']}
rows=[]
for path in sorted(data.glob('*/evaluation/report.json')):
    report=json.loads(path.read_text())
    deltas={r['frame']:r['psnr']-base_views[r['frame']]['psnr'] for r in report['test_views']}
    worst=min(deltas,key=deltas.get)
    ply=path.parent.parent/f"point_cloud/iteration_{report['iteration']}/point_cloud.ply"
    rows.append(dict(model=path.parent.parent.name,psnr=report['mean_psnr'],ssim=report['mean_ssim'],
                     depth_abs_relative=report.get('mean_depth_abs_relative'),
                     worst_psnr_delta=deltas[worst],worst_frame=worst,splats=report['splats'],
                     ply_bytes=ply.stat().st_size,chosen=path.parent.parent.name==args.chosen))
lines=['# 2DGS改善の比較記録','',f'採用モデル: `{args.chosen}`','',
       '同じ9枚の評価画像・518×294ピクセルで比較。評価画像は実写学習には使っていないが、VGGTの幾何推定には使っている。奥行きの値は実測精度ではなくVGGT推定との相対差。', '',
       '| モデル | PSNR↑ | SSIM↑ | 奥行き相対差↓ | 最も悪化した視点のPSNR差 | splats |',
       '|---|---:|---:|---:|---:|---:|']
for row in rows:
    depth='対象外' if row['depth_abs_relative'] is None else f"{row['depth_abs_relative']*100:.2f}%"
    lines.append(f"| {row['model']} | {row['psnr']:.3f} | {row['ssim']:.4f} | {depth} | {row['worst_psnr_delta']:+.3f} ({row['worst_frame']}) | {row['splats']:,} |")
lines += ['', 'カメラ補正の候補は、全画像から特徴対応を取り、3D点の位置を固定して検証。元VGGTの画素別奥行きとは視線が変わるため、奥行き相対差を直接比較しない。', '', '## 高解像度評価（1036×588、別表）', '', '| モデル | PSNR↑ | SSIM↑ |', '|---|---:|---:|']
for path in sorted(data.glob('*/evaluation-2x/report.json')):
    report=json.loads(path.read_text())
    lines.append(f"| {path.parent.parent.name} | {report['mean_psnr']:.3f} | {report['mean_ssim']:.4f} |")
(data/'experiments-summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(data/'experiments-summary.json').write_text(json.dumps(rows,indent=2))
frames=['000000','000016','000024','000048']
board=Image.new('RGB',(518*3,320*len(frames)), '#202020')
draw=ImageDraw.Draw(board)
for row,frame in enumerate(frames):
    paths=[data/'colmap/images'/f'{frame}.png',data/'train-7000/evaluation'/f'{frame}.png',data/args.chosen/'evaluation'/f'{frame}.png']
    for column,(label,path) in enumerate(zip(['Video reference','Before','After'],paths)):
        draw.text((column*518+8,row*320+6),f'{frame} / {label}',fill='white')
        with Image.open(path) as image:
            board.paste(image.resize((518,294)),(column*518,row*320+26))
board.save(data/'feedback-comparison.jpg',quality=95)
print('\n'.join(lines))
