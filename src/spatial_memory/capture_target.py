"""Validate a shooting target against the exact source model, never the latest revision."""
import json
import re
from pathlib import Path

from .capture_options import MAX_CAPTURE_PHOTOS


def resolve_capture_targets(root, raw, parent_scene_id, photo_count):
    if raw is None:
        return None
    try:
        if len(raw) > MAX_CAPTURE_PHOTOS * 2048:
            raise ValueError()
        values = json.loads(raw)
        if not isinstance(values, list) or not 1 <= len(values) <= MAX_CAPTURE_PHOTOS or len(values) != photo_count:
            raise ValueError()
        if not all(isinstance(value, dict) for value in values):
            raise ValueError()
        return [resolve_capture_target(root, json.dumps(value), parent_scene_id) for value in values]
    except (ValueError, TypeError) as error:
        raise ValueError('写真と撮影対象の組み合わせを確認できません。撮影画面からやり直してください。') from error


def resolve_capture_target(root, raw, parent_scene_id):
    if not raw:
        return None
    if len(raw) > 2048:
        raise ValueError('撮影対象の情報が大きすぎます')
    try:
        value = json.loads(raw)
        run, result, key = value['run'], value['result'], value['key']
        if not parent_scene_id or not all(isinstance(x, str) and re.fullmatch(r'[a-zA-Z0-9_-]{1,120}', x) for x in (run, result)):
            raise ValueError()
        if not isinstance(key, str) or len(key) > 160:
            raise ValueError()
        path = Path(root) / 'web/generation-reviews' / run / result / 'room.json'
        room = json.loads(path.read_text(encoding='utf-8'))
        if room.get('transform', {}).get('scene_id') != parent_scene_id:
            raise ValueError()
        surfaces = room.get('surfaces', [])
        if key.startswith('surface:'):
            selected = [s for s in surfaces if 'surface:' + str(s['id']) == key and s.get('object_id') is None and s.get('kind') in ('wall', 'floor', 'ceiling')]
            if not selected:
                raise ValueError()
            kind, object_id = selected[0]['kind'], None
        elif key.startswith('object:'):
            object_id = key[7:]
            if any(str(o['id']) == object_id and o.get('hidden') for o in room.get('furniture', [])):
                raise ValueError()
            selected = [s for s in surfaces if s.get('object_id') is not None and str(s['object_id']) == object_id]
            if not selected and not any(str(o['id']) == object_id for o in room.get('furniture', [])):
                raise ValueError()
            kind = 'object'
        else:
            raise ValueError()
        return dict(scene_id=parent_scene_id, run=run, result=result, key=key, kind=kind,
                    object_id=object_id, surface_ids=[s['id'] for s in selected],
                    status='user_selected_not_verified')
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise ValueError('撮影対象を確認できません。元の部屋から撮る場所を選び直してください。') from error
