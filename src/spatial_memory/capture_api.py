"""Capture improvement endpoints; no expensive inference in upload requests."""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError, BaseModel, Field

from .capture_options import CaptureOptions, MAX_CAPTURE_PHOTOS, MAX_CAPTURE_FRAMES
from .capture_inputs import CaptureInput, inherited_inputs, minimum_capture_frames, effective_capture_frame_budget
from .capture_target import resolve_capture_target, resolve_capture_targets


class PhotoAlignmentRequest(BaseModel):
    run: str = Field(pattern=r'^[a-zA-Z0-9_-]+$')
    result: str = Field(pattern=r'^[a-zA-Z0-9_-]+$')


def result_directory(root, job):
    query = parse_qs(urlparse(job.get('result_url') or job.get('previous_result_url', '')).query)
    run, result = query.get('run', [''])[0], query.get('result', [''])[0]
    if not run or not result or any(not all(c.isalnum() or c in '-_' for c in v) for v in (run, result)):
        return None
    return Path(root) / 'web/generation-reviews' / run / result


def parent_capture_inputs(scene, max_frames=60):
    # An unprocessed scene has no inherited observations yet.
    if scene is None or not any((scene.capture_manifest_path, scene.source_video_path, scene.frames_dir)):
        return []
    return inherited_inputs(scene, max_frames)


def capture_photo_budget(scene):
    inherited = parent_capture_inputs(scene)
    minimum = minimum_capture_frames(inherited)
    return dict(max_frames=MAX_CAPTURE_FRAMES, min_parent_frames=minimum,
                existing_photo_count=sum(item.kind == 'photo' and not item.metadata.get('reused_frame') for item in inherited),
                max_additional_photos=min(MAX_CAPTURE_PHOTOS, max(0, MAX_CAPTURE_FRAMES - minimum)), available=True)


def capture_quality(root, scene, room_jobs):
    job = room_jobs.status(scene.id)
    report = dict(schema_version=1, scene_id=scene.id, parent_scene_id=scene.parent_scene_id,
                  state=job['state'], recommendations=[], objects=[], options={'up_axis':job.get('up_axis','image-top'),**job.get('options', {})},
                  sources=[], measurements={}, result_url=job.get('result_url') or job.get('previous_result_url'))
    if not job.get('result_url') and job.get('previous_result_url'):
        report['options']={'up_axis':job.get('up_axis','image-top'),**job.get('previous_options',{})}
    report['options']=CaptureOptions.model_validate(report['options']).model_dump(mode='json')
    try:
        report['photo_budget'] = capture_photo_budget(scene)
    except (ValueError, OSError, KeyError, TypeError):
        report['photo_budget'] = dict(max_frames=MAX_CAPTURE_FRAMES, available=False)
    directory = result_directory(root, job)
    if directory and (directory / 'quality.json').is_file():
        report['appearance'] = json.loads((directory / 'quality.json').read_text(encoding='utf-8'))
        report['recommendations'].extend(report['appearance'].get('recommendations', []))
    if directory and (directory / 'room.json').is_file():
        room = json.loads((directory / 'room.json').read_text(encoding='utf-8'))
        report['calibration'] = room.get('calibration', {})
        prediction = room.get('source_prediction', '')
        if prediction and all(c.isalnum() or c in '-_' for c in prediction):
            path = Path(root) / 'data/generation-runs' / directory.parent.name / prediction / 'layout.json'
            if path.exists():
                layout = json.loads(path.read_text(encoding='utf-8'))
                report['furniture'] = layout.get('image_furniture', {}).get('quality_report', {})
                if isinstance(report['furniture'], dict):
                    report['recommendations'].extend(report['furniture'].get('recommendations', []))
                    for obj in report['furniture'].get('objects',[]):
                        if obj.get('hidden'):continue
                        for need in obj.get('needs_photo',[]):
                            report['recommendations'].append(dict(need,object_id=obj['id'],kind='furniture',severity='info'))
                for box in layout.get('boxes', []):
                    if box.get('class') == 'bbox':
                        item=dict(id=box['id'], label=box.get('label'), hidden=bool(box.get('hidden')),
                                  dimensions_model_units=box.get('scale'), evidence=box.get('evidence', {}),
                                  review_status=box.get('review_status'), source_frames=box.get('source_frames', []))
                        calibration=report.get('calibration',{})
                        if calibration.get('metric_scale_calibrated'):
                            item['dimensions_m']=[v*calibration.get('scale_factor',1) for v in box['scale']]
                        report['objects'].append(item)
    if scene.capture_manifest_path and Path(scene.capture_manifest_path).is_file():
        capture = json.loads(Path(scene.capture_manifest_path).read_text(encoding='utf-8'))
        report['capture'] = {key: value for key, value in capture.items()
                             if key in ('limitations', 'quality', 'selection', 'parent_scene_id', 'options', 'duplicates_removed')}
    if scene.frames_dir:
        manifest = Path(scene.frames_dir) / 'manifest.json'
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding='utf-8'))
            report['frame_count'] = len(data.get('frames', []))
            report['frames'] = [dict(index=i, url=f'/scenes/{scene.id}/frames/{i}',
                                    kind=frame.get('source_kind', frame.get('kind', 'video')),
                                    timestamp_seconds=frame.get('timestamp_seconds'))
                                for i, frame in enumerate(data.get('frames', []))]
    if scene.reconstruction_npz:
        metrics_path=Path(scene.reconstruction_npz).parent/'metrics.json'
        if metrics_path.is_file():
            metrics=json.loads(metrics_path.read_text(encoding='utf-8'))
            if 'camera_refinement' in metrics:
                report['camera_refinement']=metrics['camera_refinement']
    if not report['recommendations']:
        report['recommendations'].append(dict(kind='capture', severity='info', code='check_views',
            message='家具の正面・側面と、床・天井・部屋の角が写っているか確認してください。未観測の裏側は復元できません。'))
    return report


def build_capture_router(root, db, capture_jobs, room_jobs):
    router = APIRouter()

    @router.post('/capture-jobs')
    def start_capture(name: str = Form('新しい部屋'), parent_scene_id: str | None = Form(None),
                      capture_target: str | None = Form(None), capture_targets: str | None = Form(None), options: str = Form('{}'), videos: list[UploadFile] | None = File(None),
                      photos: list[UploadFile] | None = File(None),
                      panoramas: list[UploadFile] | None = File(None),
                      depth_bundle: UploadFile | None = File(None)):
        try:
            settings = CaptureOptions.model_validate_json(options).model_dump(mode='json')
            if len(name) > 120 or not name.strip():
                raise ValueError('部屋名は1～120文字で入力してください')
            if parent_scene_id:
                parent_scene_id = str(UUID(parent_scene_id))
                if not db.get_scene(parent_scene_id):
                    raise ValueError('追加先の部屋が見つかりません')
            if capture_target is not None and capture_targets is not None:
                raise ValueError('撮影対象の指定が重複しています')
            target_context = resolve_capture_target(root, capture_target, parent_scene_id)
            photo_targets = resolve_capture_targets(root, capture_targets, parent_scene_id, len(photos or []))
            if (target_context or photo_targets) and (not photos or videos or panoramas or depth_bundle):
                raise ValueError('撮影対象を指定する場合は、写真だけを追加してください')
            groups = [('video', videos or [], 6), ('photo', photos or [], MAX_CAPTURE_PHOTOS),
                      ('panorama', panoramas or [], 4), ('sensor', [depth_bundle] if depth_bundle else [], 1)]
            if not any(files for _, files, _ in groups):
                raise ValueError('動画・写真・パノラマ・深度データのいずれかを選んでください')
            if depth_bundle and (parent_scene_id or any((videos, photos, panoramas))):
                raise ValueError('深度付き撮影データは単独で新しい部屋として取り込んでください')
            for kind, files, limit in groups:
                if len(files) > limit:
                    raise ValueError(f'{kind}は最大{limit}個までです')
            if not depth_bundle:
                parent = db.get_scene(parent_scene_id) if parent_scene_id else None
                inherited = parent_capture_inputs(parent, settings['max_frames'])
                incoming = [CaptureInput(Path(upload.filename or 'capture'), kind)
                            for kind, files, _ in groups for upload in files]
                settings['max_frames'] = effective_capture_frame_budget(inherited + incoming, settings['max_frames'])
            if capture_jobs.active or room_jobs.active:
                raise RuntimeError('別の部屋を処理中です。完了後に追加できます。')
            identifier = str(uuid4())
            destination = capture_jobs.folder(identifier) / 'uploads'
            destination.mkdir(parents=True)
            inputs, total = [], 0
            for kind, files, limit in groups:
                if len(files) > limit:
                    raise ValueError(f'{kind}は最大{limit}個までです')
                for upload_index, upload in enumerate(files):
                    extension = Path(upload.filename or '').suffix.lower()
                    supported = {'video': {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'},
                                 'photo': {'.jpg', '.jpeg', '.png', '.webp'},
                                 'panorama': {'.jpg', '.jpeg', '.png', '.webp'}, 'sensor': {'.zip'}}[kind]
                    if extension not in supported:
                        raise ValueError(f'対応していないファイル形式です: {extension}')
                    path = destination / (uuid4().hex + extension)
                    size = 0
                    with path.open('xb') as target:
                        while chunk := upload.file.read(1024 * 1024):
                            size += len(chunk)
                            total += len(chunk)
                            if size > (256 if kind in ('video', 'sensor') else 64) * 1024**2 or total > 512 * 1024**2:
                                raise ValueError('ファイルが大きすぎます。短く分けた動画や写真で追加してください。')
                            target.write(chunk)
                    if not size:
                        raise ValueError('空のファイルは使えません')
                    metadata = {'original_name': Path(upload.filename).name}
                    if kind == 'photo' and (target_context or photo_targets):
                        metadata['capture_target'] = photo_targets[upload_index] if photo_targets else target_context
                    inputs.append(dict(kind=kind, path=str(path), metadata=metadata))
            result = capture_jobs.start(inputs, name.strip(), parent_scene_id=parent_scene_id,
                                        options=settings, job_id=identifier)
            return capture_jobs.public(result)
        except (ValueError, ValidationError) as error:
            raise HTTPException(422, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @router.get('/capture-jobs/{job_id}')
    def capture_status(job_id: str):
        try:
            return capture_jobs.public(capture_jobs.status(job_id))
        except ValueError as error:
            raise HTTPException(422, '処理IDが不正です') from error
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error

    @router.post('/capture-jobs/{job_id}/resume')
    def capture_resume(job_id: str):
        try:
            return capture_jobs.public(capture_jobs.resume(job_id))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @router.get('/scenes/{scene_id}/capture-quality')
    def quality(scene_id: str):
        scene = db.get_scene(scene_id)
        if not scene:
            raise HTTPException(404, '部屋が見つかりません')
        try:
            return capture_quality(root, scene, room_jobs)
        except (ValueError, KeyError) as error:
            raise HTTPException(422, '部屋の品質記録を読み取れませんでした') from error

    @router.post('/scenes/{scene_id}/photo-alignment')
    def photo_alignment(scene_id: str, request: PhotoAlignmentRequest):
        if not db.get_scene(scene_id):raise HTTPException(404, '部屋が見つかりません')
        if capture_jobs.active:raise HTTPException(409, '追加撮影の完了後にお試しください')
        try:
            previous=room_jobs.status(scene_id)
            if previous['state']=='running':raise RuntimeError('部屋を処理中です。完了までお待ちください')
            directory=result_directory(root,previous)
            if not directory or directory.parent.name!=request.run or directory.name!=request.result:
                raise RuntimeError('新しい生成結果があります。部屋一覧から最新の部屋を開いてください')
            room=json.loads((directory/'room.json').read_text(encoding='utf-8'))
            if room.get('photo_alignment'):raise RuntimeError('この結果は補正済みです。比較画面で結果を確認してください')
            source=room.get('source_prediction')
            if not source:raise ValueError('写真の解析結果が見つかりません')
            options=CaptureOptions.model_validate(previous.get('options') or previous.get('previous_options') or {})
            return room_jobs.start(scene_id,options.up_axis,force=True,options=options.model_dump(mode='json'),
                reuse_prediction=source,reuse_run=request.run,align_photos=True)
        except (ValueError,FileNotFoundError) as error:raise HTTPException(422,str(error)) from error
        except RuntimeError as error:raise HTTPException(409,str(error)) from error

    @router.post('/scenes/{scene_id}/capture-review')
    def review(scene_id: str, options: CaptureOptions):
        if not db.get_scene(scene_id):
            raise HTTPException(404, '部屋が見つかりません')
        if capture_jobs.active:
            raise HTTPException(409, '追加撮影を処理中です。完了後に確認してください。')
        try:
            previous = room_jobs.status(scene_id)
            if previous['state']=='running':raise RuntimeError('部屋を処理中です。完了後に確認してください。')
            directory=result_directory(root,previous)
            if not directory or not (directory/'room.json').is_file():
                raise ValueError('先に写真付き部屋モデルを完成させてください')
            saved=json.loads((directory/'room.json').read_text(encoding='utf-8'))
            source=saved.get('source_prediction')
            if not source:raise ValueError('確認元の家具推定を読み取れません')
            if options.up_axis!=saved.get('transform',{}).get('camera_up_axis',previous.get('up_axis','image-top')):
                raise ValueError('家具の確認では動画の上下方向を変更できません。元の向きに戻して更新してください。')
            baseline=previous.get('options',{}) if previous.get('result_url') else previous.get('previous_options',{})
            baseline=CaptureOptions.model_validate(baseline)
            if any(getattr(options,key)!=getattr(baseline,key) for key in ('max_frames','furniture_frame_budget','refine_cameras')):
                raise ValueError('画像枚数・カメラ位置調整は撮影データから新しく作成する際の設定です。家具の確認では元の設定に戻してください。')
            return room_jobs.start(scene_id, options.up_axis, force=True,
                                   options=options.model_dump(mode='json'), reuse_prediction=source,reuse_run=directory.parent.name)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    return router
