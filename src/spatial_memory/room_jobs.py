"""Local asynchronous layout/photo jobs. Never modify reconstruction inputs."""
import json
import hashlib
import subprocess
import sys
import threading
import shutil
from pathlib import Path
from uuid import UUID, uuid4

PIPELINE_VERSION = 5


def copy_prediction_masks(source, destination, layout):
    """A reviewed layout keeps the exact observation assets its identities refer to."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    copied = set()
    for detection in layout.get('image_furniture', {}).get('detections', []):
        relative = detection.get('mask')
        if not relative or relative in copied:
            continue
        old = (source / relative).resolve()
        new = (destination / relative).resolve()
        if not old.is_relative_to(source) or not new.is_relative_to(destination):
            raise ValueError('Observation mask path escapes prediction directory')
        if not old.is_file():
            if layout.get('image_furniture', {}).get('version', 0) >= 3:
                raise ValueError('Source observation mask is missing')
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, new)
        copied.add(relative)
    if layout.get('photo_alignment'):
        camera=source/'alignment-cameras.npz'
        if hashlib.sha256(camera.read_bytes()).hexdigest()!=layout['photo_alignment']['camera_sha256']:
            raise ValueError('Aligned camera evidence changed')
        shutil.copy2(camera,destination/camera.name)
        for name in ('alignment-before.json','alignment.json'):
            if (source/name).is_file():shutil.copy2(source/name,destination/name)


class RoomJobs:
    def __init__(self, root, scene_root):
        self.root, self.scene_root = root, scene_root
        self.directory = root/'data/room-jobs'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.active = None
        self.capture_jobs = None

    def path(self, scene):
        return self.directory/f'{UUID(scene)}.json'

    def status(self, scene):
        path = self.path(scene)
        if not path.exists():return dict(state='idle')
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['state']=='running' and self.active != scene:
            result.update(state='interrupted',message='処理が中断されました。ログを残して別の実行として再開できます。')
        return result

    def save(self, scene, result):
        path = self.path(scene);temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False),encoding='utf-8');temp.replace(path)

    def start(self, scene, up_axis='image-top', force=False, options=None, reuse_prediction=None,reuse_run=None,_capture_job_id=None,align_photos=False):
        if up_axis not in ('image-top','image-left','image-right','image-bottom'):raise ValueError('画像の上下方向が不正です')
        scene = str(UUID(scene))
        with self.lock:
            if self.capture_jobs and self.capture_jobs.active and self.capture_jobs.active!=_capture_job_id:
                raise RuntimeError('追加撮影を処理中です。完了後に開始してください。')
            previous=self.status(scene)
            if previous['state']=='running' or (previous['state']=='complete' and previous.get('pipeline_version',0)>=PIPELINE_VERSION and not force and options is None):return previous
            if self.active:raise RuntimeError('別の部屋を処理中です。完了後に試してください。')
            source=self.scene_root/scene
            for relative in ('reconstruction/reconstruction.npz','reconstruction/metrics.json','spatial_model/manifest.json','frames/manifest.json'):
                if not (source/relative).is_file():raise ValueError('先に動画から3Dシーンを作成してください。')
            metrics=json.loads((source/'reconstruction/metrics.json').read_text(encoding='utf-8'))
            if metrics.get('model') not in ('VGGT-1B','calibrated-rgbd') or metrics.get('fallback'):raise ValueError('VGGTまたは深度付き撮影で再構成された部屋が必要です。')
            run='room-'+uuid4().hex[:16]
            digest=hashlib.sha256((source/'reconstruction/reconstruction.npz').read_bytes()).hexdigest()
            for transform in sorted((self.root/'data/generation-runs').glob('room-*/transform.json'),key=lambda p:p.stat().st_mtime,reverse=True):
                saved=json.loads(transform.read_text(encoding='utf-8'))
                if saved.get('scene_id')==scene and saved.get('camera_up_axis')==up_axis and saved.get('source_npz_sha256')==digest and (transform.parent/'prediction-001/layout.json').is_file():
                    run=transform.parent.name;break
            if reuse_prediction:
                if not reuse_run or not all(c.isalnum() or c in '-_' for c in reuse_run):
                    raise ValueError('確認元の部屋データが不正です')
                source_transform=self.root/'data/generation-runs'/reuse_run/'transform.json'
                if not source_transform.is_file():raise ValueError('確認元の部屋データが見つかりません')
                original=json.loads(source_transform.read_text(encoding='utf-8'))
                if original.get('scene_id')!=scene or original.get('source_npz_sha256')!=digest or original.get('camera_up_axis')!=up_axis:
                    raise ValueError('確認元の部屋と座標・向きが一致しません')
                run=reuse_run
            if previous['state']!='idle':
                archive=self.directory/'history';archive.mkdir(exist_ok=True)
                (archive/f'{scene}-{uuid4().hex}.json').write_text(json.dumps(previous,ensure_ascii=False),encoding='utf-8')
            if reuse_prediction:
                if not all(c.isalnum() or c in '-_' for c in reuse_prediction) or not (self.root/'data/generation-runs'/run/reuse_prediction/'layout.json').is_file():
                    raise ValueError('確認元の家具推定が見つかりません')
            # Keep capture/measurement preferences on regeneration. Object IDs
            # are regenerated, so old per-object overrides cannot be transferred.
            if options is None and previous.get('options'):
                options = dict(previous['options'])
                if not reuse_prediction:
                    options['furniture_overrides'] = []
                options['up_axis'] = up_axis
            job=dict(state='running',run=run,scene=scene,up_axis=up_axis,pipeline_version=PIPELINE_VERSION,
                     image_prediction='prediction-image-'+uuid4().hex[:8],message='処理を準備しています')
            if options is not None:
                from .capture_options import CaptureOptions
                job['options']=CaptureOptions.model_validate(options).model_dump(mode='json',exclude_none=True)
            if reuse_prediction:job['reuse_prediction']=reuse_prediction
            if align_photos:job['align_photos']=True
            if previous.get('result_url') or previous.get('previous_result_url'):
                job['previous_result_url']=previous.get('result_url') or previous['previous_result_url']
                job['previous_options']=previous.get('options',{}) if previous.get('result_url') else previous.get('previous_options',{})
            self.active=scene;self.save(scene,job)
            threading.Thread(target=self.work,args=(scene,job),daemon=True).start()
            return job

    def work(self, scene, job):
        run=self.root/'data/generation-runs'/job['run']
        def step(message, script, *args):
            job.update(message=message);self.save(scene,job)
            with (self.directory/f'{job["run"]}.log').open('a',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(self.root/'scripts'/script),*map(str,args)],cwd=self.root,
                               stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            option_args=[]
            if 'options' in job:
                options_path=self.directory/f'{scene}-{job["image_prediction"]}-options.json'
                options_path.write_text(json.dumps(job['options'],ensure_ascii=False),encoding='utf-8')
                option_args=['--options',str(options_path)]
            if not (run/'transform.json').is_file():
                step('1/6 元動画の形状をAI用に準備中','prepare_spatiallm.py',self.scene_root/scene,run,'--up-axis',job.get('up_axis','image-top'))
            if not (run/'prediction-001/layout.json').is_file():
                step('2/6 AIが壁・窓の配置を推定中','run_spatiallm.py',run,'--attempt','1')
            if not (run/'prediction-auto-001/layout.json').is_file():
                step('3/6 床・天井の高さを確認中','assemble_spatiallm.py',run,'--architecture','prediction-001','--objects','prediction-001','--output','prediction-auto-001','--allow-assumed-ceiling')
            # No scene-specific category selection or manual corrections are reused.
            prediction=job.setdefault('image_prediction','prediction-image-'+uuid4().hex[:8])
            if job.get('reuse_prediction'):
                from .furniture_quality import apply_furniture_overrides
                from .texture_quality import calibration_scale
                original=json.loads((run/job['reuse_prediction']/'layout.json').read_text(encoding='utf-8'))
                updated=apply_furniture_overrides(original,job.get('options',{}).get('furniture_overrides',[]),
                    unit_scale=calibration_scale(original,job.get('options',{})))
                revised=run/prediction;revised.mkdir(exist_ok=False)
                copy_prediction_masks(run/job['reuse_prediction'], revised, updated)
                if updated.get('photo_alignment'):
                    web_root=self.root/'web/generation-reviews'/run.name
                    shutil.copytree(web_root/job['reuse_prediction']/'alignment',web_root/prediction/'alignment')
                (revised/'layout.json').write_text(json.dumps(updated,ensure_ascii=False),encoding='utf-8')
            else:
                step('4/6 写真の鮮明さと視点を選び、家具を複数画像で確認中','infer_image_furniture.py',run,*option_args,'--output',prediction)
            if not (run/prediction/'layout.json').is_file():raise RuntimeError('画像からの家具推定結果が保存されていません')
            if job.get('align_photos'):
                step('元写真と照合し、カメラ・床・壁のずれを補正中','align_room_photos.py',run,'--prediction',prediction,'--before-url',job.get('previous_result_url',''))
            step('5/6 元写真と推定位置の比較を作成中','review_spatiallm.py',run,'--prediction',prediction)
            job['review_url']=f'/static/generation-review.html?run={run.name}&prediction={prediction}';self.save(scene,job)
            output='solid-photo-'+uuid4().hex[:8]
            step('6/6 部屋と家具に写真を貼り、寸法・写真の不足・歩行範囲を確認中','build_solid_room.py','--run',run.name,'--prediction',prediction,*option_args,'--output',output)
            result_dir=self.root/'web/generation-reviews'/run.name/output
            room=json.loads((result_dir/'room.json').read_text(encoding='utf-8'))
            if not room.get('geometry_validation',{}).get('shell_watertight') or not room.get('navigation') or not (result_dir/'room.glb').is_file():
                raise RuntimeError('閉じた部屋モデルの検証が完了していません')
            job.update(state='complete',message='写真を貼った部屋モデルが完成しました。壁・家具との衝突判定付きで歩けます。家具や未観測部分は簡易形状です。',
                       result_url=f'/static/solid-room.html?run={run.name}&result={output}',
                       model_url=f'/static/generation-reviews/{run.name}/{output}/room.glb',
                       quality_url=f'/scenes/{scene}/capture-quality')
        except Exception as error:
            job.update(state='failed',message='自動処理を完了できませんでした。推定の比較結果があれば確認してください。',error=str(error),log=f'{job["run"]}.log')
            log=self.directory/f'{job["run"]}.log'
            if log.exists() and 'Camera up directions are ambiguous' in log.read_text(encoding='utf-8',errors='replace'):
                job['message']='動画の上下方向を判定できません。元画像で部屋の上がどちらか選び、再実行してください。'
            elif log.exists() and any(text in log.read_text(encoding='utf-8',errors='replace') for text in ('perimeter','Room frame needs review','closed consistent volume')):
                job['message']='床や壁を閉じた部屋として確定できませんでした。比較結果を確認してください。部屋の角を重ねて写した動画で再試行できます。'
        finally:
            self.save(scene,job)
            with self.lock:self.active=None
