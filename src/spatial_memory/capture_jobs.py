"""Durable capture revisions. Uploaded sources and older scenes are never replaced."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import threading
import time
from uuid import UUID, uuid4

from .capture_options import CaptureOptions


class CaptureJobs:
    def __init__(self, root, reconstruction, room_jobs):
        self.directory = Path(root) / 'data' / 'capture-jobs'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.reconstruction = reconstruction
        self.room_jobs = room_jobs
        self.lock = getattr(room_jobs, 'lock', threading.RLock())
        room_jobs.capture_jobs = self
        self.active = None

    @contextmanager
    def legacy_reconstruction(self):
        with self.lock:
            if self.active or self.room_jobs.active:
                raise RuntimeError('別の部屋を処理中です。完了後に開始してください。')
            self.active='legacy-reconstruction'
        try:
            yield
        finally:
            with self.lock:self.active=None

    def folder(self, job_id):
        return self.directory / str(UUID(str(job_id)))

    def save(self, job):
        folder = self.folder(job['id'])
        folder.mkdir(parents=True, exist_ok=True)
        with self.lock:
            temporary = folder / f'job-{uuid4().hex}.tmp'
            try:
                temporary.write_text(json.dumps(job, ensure_ascii=False, allow_nan=False), encoding='utf-8')
                for attempt in range(8):
                    try:
                        temporary.replace(folder / 'job.json')
                        break
                    except PermissionError:
                        if attempt == 7:
                            raise
                        time.sleep(.05 * (attempt + 1))
            finally:
                temporary.unlink(missing_ok=True)

    def status(self, job_id):
        path = self.folder(job_id) / 'job.json'
        if not path.is_file():
            raise FileNotFoundError('撮影データの処理が見つかりません')
        with self.lock:
            job = json.loads(path.read_text(encoding='utf-8'))
        if job.get('stage') == 'room' and job.get('scene_id') and job['state'] != 'complete':
            room = self.room_jobs.status(job['scene_id'])
            if room.get('state') == 'complete' and room.get('result_url'):
                job.update(state='complete', stage='complete', message=room['message'],
                           result_url=room['result_url'], review_url=room.get('review_url'),
                           quality_url=f"/scenes/{job['scene_id']}/capture-quality")
                job.pop('error', None)
                self.save(job)
        if job['state'] in ('running', 'queued') and job['id'] != self.active:
            job.update(state='interrupted', message='サーバーの再起動で中断しました。アップロードした素材は保存されています。再開できます。')
        return job

    def start(self, inputs, name, *, parent_scene_id=None, options=None, job_id=None):
        options = CaptureOptions.model_validate(options or {}).model_dump(mode='json')
        if parent_scene_id is not None:
            parent_scene_id = str(UUID(parent_scene_id))
            if not self.reconstruction.db.get_scene(parent_scene_id):
                raise ValueError('追加先の部屋が見つかりません')
        with self.lock:
            if self.active or self.room_jobs.active:
                raise RuntimeError('別の部屋を処理中です。完了してから開始してください。')
            identifier = str(UUID(job_id)) if job_id else str(uuid4())
            if (self.folder(identifier) / 'job.json').exists():
                raise ValueError('このアップロードはすでに処理を開始しています')
            job = dict(id=identifier, state='queued', stage='capture', name=name,
                       parent_scene_id=parent_scene_id, options=options, inputs=inputs,
                       created_at=time.time(), message='撮影データを受け取りました。全体から必要な視点を選びます。')
            self.active = identifier
            self.save(job)
            threading.Thread(target=self.work, args=(job,), daemon=True).start()
            return job

    def resume(self, job_id):
        with self.lock:
            job = self.status(job_id)
            if job['state'] == 'complete' or job_id == self.active:
                return job
            if self.active or self.room_jobs.active:
                raise RuntimeError('別の部屋を処理中です。完了後に再開してください。')
            self.active = job['id']
            job.update(state='queued', message='保存済みの素材から再開します。')
            self.save(job)
            threading.Thread(target=self.work, args=(job,), daemon=True).start()
            return job

    def work(self, job):
        try:
            job.update(state='running', message='動画全体・追加写真から、同じ座標の部屋形状を推定しています。')
            self.save(job)
            # Resume the model stage without rerunning successful reconstruction.
            if not job.get('scene_id'):
                inputs = [dict(item, path=Path(item['path'])) for item in job['inputs']]
                sensor = [item for item in inputs if item['kind'] == 'sensor']
                if sensor:
                    if len(inputs) != 1 or job.get('parent_scene_id'):
                        raise ValueError('深度付き撮影データは単独で新しい部屋として取り込んでください。')
                    scene, metrics = self.reconstruction.reconstruct_sensor_capture(
                        sensor[0]['path'], job['name'], max_frames=job['options']['max_frames'], capture_metadata=job['options'])
                else:
                    scene, metrics = self.reconstruction.reconstruct_capture(
                        inputs, job['name'], parent_scene_id=job.get('parent_scene_id'),
                        max_frames=job['options']['max_frames'], sample_fps=4,
                        quality_filter=True, capture_metadata=job['options'])
                job.update(scene_id=scene.id, metrics=metrics, stage='room')
                self.save(job)
            options = dict(job['options'])
            # Furniture identifiers belong to a particular inference result. Never
            # carry a correction to an unrelated re-estimated object by coincidence.
            options['furniture_overrides'] = []
            room = self.room_jobs.start(job['scene_id'], options['up_axis'], options=options,_capture_job_id=job['id'])
            while room['state'] == 'running':
                job.update(message=room.get('message', '部屋を作成しています'), room_state=room['state'])
                self.save(job)
                time.sleep(1)
                room = self.room_jobs.status(job['scene_id'])
            if room['state'] != 'complete':
                raise RuntimeError(room.get('message', '部屋モデルを完成できませんでした'))
            job.update(state='complete', stage='complete', message=room['message'],
                       result_url=room['result_url'], review_url=room.get('review_url'),
                       quality_url=f'/scenes/{job["scene_id"]}/capture-quality')
        except Exception as error:
            job.update(state='failed', message=str(error), error=str(error))
        finally:
            try:
                self.save(job)
            finally:
                with self.lock:
                    self.active = None

    @staticmethod
    def public(job):
        # Source locations are implementation details, not a public API contract.
        return {key: value for key, value in job.items() if key not in ('inputs', 'metrics')}
