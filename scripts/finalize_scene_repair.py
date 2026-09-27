"""Quantitative/visual before-after review; never overwrite the active model."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from spatial_memory.scene_repair import repair_decision
from PIL import Image,ImageDraw

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--best-ply',type=Path,required=True)
p.add_argument('--geometry',type=Path,required=True);a=p.parse_args()
run=a.run.resolve()
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
before=read(run/'best-evaluation/report.json');after=read(run/'candidate-evaluation/report.json')
control=read(run/'control-evaluation/report.json')
repair=read(run/'repair-colmap/repair.json');training=read(run/'candidate/repair-training.json')
decision=repair_decision(before,after,repair['consistency'])
advantage=dict(psnr=after['mean_psnr']-control['mean_psnr'],ssim=after['mean_ssim']-control['mean_ssim'])
decision['versus_no_ai_control']=advantage
if advantage['psnr']<.01 and advantage['ssim']<.0002:
    decision['accepted']=False;decision['reasons'].append('No measurable benefit versus no-AI distillation control')
if not training['geometry_and_opacity_bit_exact']:
    decision['accepted']=False;decision['reasons'].append('Geometry freeze failed')
source_hash=hashlib.sha256(a.best_ply.read_bytes()).hexdigest()
geometry_hash=hashlib.sha256(a.geometry.read_bytes()).hexdigest()
if source_hash!=repair['source_ply_sha256'] or geometry_hash!=repair['source_npz_sha256']:
    raise ValueError('Original inputs changed')
decision.update(current_best_preserved=True,geometry_preserved=True,registered=False,
                model_status='accepted_candidate' if decision['accepted'] else 'rejected_experiment',
                lpips='not measured in this prototype',
                unknown_completion='not attempted',repair_pixels=repair['consistency']['changed_views'])
(run/'decision.json').write_text(json.dumps(decision,indent=2))
name='accepted-candidate.json' if decision['accepted'] else 'rejected-experiment.json'
(run/name).write_text(json.dumps(dict(model=str(run/'candidate'),decision=decision),indent=2))
views=[r['frame'] for r in before['test_views']]
w,h=before['image_size']
board=Image.new('RGB',(w*4,(h+28)*len(views)),'#202020');draw=ImageDraw.Draw(board)
for row,frame in enumerate(views):
    for column,(label,path) in enumerate([
        ('Real held-out photo',run/'snapshot'/frame/'observed.png'),
        ('Current best',run/'best-evaluation'/f'{frame}.png'),
        ('No-AI control',run/'control-evaluation'/f'{frame}.png'),
        ('AI repair candidate',run/'candidate-evaluation'/f'{frame}.png')]):
        draw.text((column*w+8,row*(h+28)+6),f'{frame} / {label}',fill='white')
        with Image.open(path) as im: board.paste(im,(column*w,row*(h+28)+28))
board.save(run/'heldout-before-after.jpg',quality=95)
# Show the actual masked repair targets at high magnification, not the full generated proposal.
board=Image.new('RGB',(5*256,3*280),'#202020');draw=ImageDraw.Draw(board)
for row,changed in enumerate(repair['consistency']['changed_views']):
    frame=changed['frame']
    with Image.open(run/'repair-colmap/provenance'/f'{frame}.png') as im:
        import numpy as np
        yy,xx=np.where(np.asarray(im)==3)
    cx,cy=int(xx.mean()),int(yy.mean()); box=(cx-32,cy-32,cx+32,cy+32)
    for column,(label,path) in enumerate([
        ('Observed',run/'snapshot'/frame/'observed.png'),('Before',run/'snapshot'/frame/'render.png'),
        ('AI training target',run/'repair-colmap/images'/f'{frame}.png'),
        ('After training',run/'candidate-evaluation/repair-views'/f'{frame}.png'),
        ('Approved pixels',run/'repair-colmap/provenance'/f'{frame}.png')]):
        with Image.open(path) as im:
            if column==4: im=im.point(lambda x:255 if x==3 else 0).convert('RGB')
            board.paste(im.crop(box).resize((256,256),Image.Resampling.NEAREST),(column*256,row*280+24))
        draw.text((column*256+5,row*280+4),f'{frame} / {label}',fill='white')
board.save(run/'repair-patches.png')
status='候補として採用（未登録）' if decision['accepted'] else '不採用：現在のモデルを維持'
table=''.join(f"<tr><td>{label}</td><td>{r['mean_psnr']:.5f}</td><td>{r['mean_ssim']:.6f}</td><td>{r['mean_depth_to_best_relative']:.8f}</td><td>{r['mean_hole_fraction']:.6f}</td><td>{r['repair_multiview_mae']:.6f}</td></tr>"
              for label,r in [('現在のベスト',before),('AIなし追加学習',control),('AI修復後',after)])
page=f'''<!doctype html><meta charset="utf-8"><title>AI修復実験の比較</title>
<style>body{{background:#121820;color:#e9eef5;font:16px system-ui;max-width:1600px;margin:32px auto;padding:20px}}img{{max-width:100%}}pre{{white-space:pre-wrap}}a{{color:#8bcaff}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #456}}</style>
<h1>AI修復実験：{status}</h1><p>修復は観測・奥行きの裏付けがある3視点の小領域のみ。形状と透明度は固定。大きな未観測領域の補完は未実施です。</p>
<p>AI生成の修復案を同じ3D位置に投影しています。整合性チェックは物理的な正しさの保証ではありません。評価画像は外観学習から除外しましたが、VGGTとカメラ補正では使用しています。</p>
<h2>定量比較</h2><table><tr><th>モデル</th><th>PSNR↑</th><th>SSIM↑</th><th>元モデルからの深度変化↓</th><th>低透明度の領域率↓</th><th>別視点との色の差↓</th></tr>{table}</table>
<p>深度と欠損は実測ではなくモデル同士の比較です。LPIPSは未測定。AI修復なしの対照よりも改善し、平均・最悪視点・深度・複数視点の条件をすべて満たした候補だけ採用します。</p>
<details><summary>判定の詳細</summary><pre>{html.escape(json.dumps(decision,ensure_ascii=False,indent=2))}</pre></details>
<h2>実際に変更した学習画素（拡大）</h2><img src="repair-patches.png">
<h2>実写／現在のベスト／AIなし対照／修復後候補：評価用9視点</h2><img src="heldout-before-after.jpg">'''
(run/'review.html').write_text(page,encoding='utf-8')
print(json.dumps(decision,indent=2))
