"""Bake a separate non-point-cloud preview from AI boxes and recorded photographs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import cv2
from PIL import Image
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from spatial_memory.solid_room import wall_loop, photo_visibility, closed_shell, furniture_parts, triangulate_polygon, navigation_spawn, furniture_occlusion
from spatial_memory.texture_quality import (PhotoCache, sample_photo,
    estimate_photo_gains, measurement_calibration, scale_room_report, capture_quality_report)
from spatial_memory.wall_evidence import depth_normals, matching_normals, independent_support
from spatial_memory.furniture_quality import camera_viewpoints
from spatial_memory.material_completion import complete_materials
from spatial_memory.regional_textures import regional_face_texture
from spatial_memory.photo_extension import horizontal_photo_regions, extend_same_photos
from spatial_memory.semantic_textures import SemanticPhotoMasks, select_face_photos, coherent_face_texture


def build(run, prediction, output, options=None):
    options = options or {}
    source = run/prediction/'layout.json'
    layout = json.loads(source.read_text(encoding='utf-8'))
    requested_overrides = options.get('furniture_overrides') or []
    if requested_overrides and layout.get('furniture_overrides') != requested_overrides:
        raise ValueError('Furniture overrides must be applied to the prediction layout before texture export')
    transform = json.loads((run/'transform.json').read_text(encoding='utf-8'))
    scene = ROOT/'data/scenes-v2'/transform['scene_id']
    reconstruction = scene/'reconstruction/reconstruction.npz'
    metrics_path = reconstruction.with_name('metrics.json')
    reconstruction_metrics = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else {}
    sensor_camera = reconstruction_metrics.get('model') == 'calibrated-rgbd'
    with reconstruction.open('rb') as evidence:
        if hashlib.file_digest(evidence, 'sha256').hexdigest() != transform['source_npz_sha256']:
            raise ValueError('Source evidence changed')
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    # Zero-thickness architecture fails the volumetric evidence flag by design.
    # Keep those inferred planes; photo support is independently checked below.
    boxes = [b for b in layout['boxes'] if not b.get('hidden') and (b['class'] == 'wall' or b['evidence']['valid'])]
    footprint = wall_loop(boxes)
    floor = layout['vertical_anchors']['floor']['z']
    ceiling = layout['vertical_anchors']['ceiling']['z']
    calibration = measurement_calibration(footprint, floor, ceiling, options)
    if sensor_camera and calibration['primary_measurement'] is None:
        calibration.update(metric_scale_calibrated=True, source='sensor_calibration',
            assumptions=['Metre scale supplied by the RGB-D camera calibration; scene geometry is not independently verified.'])
    output_scale = calibration['scale_factor']
    shell_vertices, shell_faces = closed_shell(footprint, floor, ceiling)
    shell = trimesh.Trimesh(shell_vertices, shell_faces, process=True)
    if not shell.is_watertight or not shell.is_winding_consistent or shell.volume <= 0:
        raise ValueError('Room shell is not a closed consistent volume')
    with np.load(reconstruction, allow_pickle=False) as saved:
        depths = saved['depth'].copy().squeeze()
        confidences = saved['confidence'].copy().squeeze()
        es, ks = saved['extrinsics'].copy(), saved['intrinsics'].copy()
    alignment = layout.get('photo_alignment')
    if alignment:
        camera_path = run/prediction/'alignment-cameras.npz'
        if alignment['source_npz_sha256'] != transform['source_npz_sha256'] or hashlib.sha256(camera_path.read_bytes()).hexdigest() != alignment['camera_sha256']:
            raise ValueError('Alignment evidence changed')
        with np.load(camera_path,allow_pickle=False) as adjusted:
            corrected=adjusted['extrinsics']
            if corrected.shape != es.shape or not np.isfinite(corrected).all():raise ValueError('Invalid aligned cameras')
            es=corrected.copy()
    records = json.loads((scene/'frames/manifest.json').read_text(encoding='utf-8'))['frames']
    n, h, w = depths.shape
    if len(records) != n:
        raise ValueError('Frame count mismatch')
    semantic_masks = SemanticPhotoMasks(run/prediction, layout, h, w)
    inverse = np.asarray(transform['generation_world_to_vg_world'])
    world_to_room=np.asarray(transform['vg_world_to_generation_world'])
    texture_blockers=[b for b in boxes if b['class']=='bbox']
    scale = np.linalg.norm(np.asarray(transform['vg_world_to_generation_world'])[:3, 0])
    projection_tolerance = .14 / (scale * output_scale)
    high_quality = options.get('texture_quality') == 'high'
    photos = PhotoCache(scene/'frames', records, h, w, max_scale=4 if high_quality else 3,
                       preprocessing='aligned_rgbd' if sensor_camera else 'vggt')
    gains = estimate_photo_gains(photos, es, ks, depths, confidences, projection_tolerance)
    sharpness = np.asarray([max(0, record.get('sharpness') or 0) for record in records], float)
    sharpness = np.where(sharpness > 0, .65 + .35 * sharpness / max(float(sharpness.max()), 1), .8)
    horizontal_regions=horizontal_photo_regions(depths,es,ks,world_to_room)
    extension_tolerance=.35/(scale*output_scale)
    normals = depth_normals(depths, es, ks)
    viewpoint_groups, _ = camera_viewpoints(records, es)
    output.mkdir(parents=True)
    surfaces, models, colliders = [], [], []
    exported = trimesh.Scene()

    def surface(name, origin, u, v, color, kind, polygon=None, object_box=None, outward=None):
        origin, u, v = map(lambda a: np.asarray(a, dtype=float), (origin, u, v))
        tw, th = [int(np.clip(round(np.linalg.norm(a)*output_scale*160), 32, 1536 if high_quality else 1024)) for a in (u, v)]
        # Image top corresponds to +v; UV bottom corresponds to origin.
        xx, yy = np.meshgrid((np.arange(tw)+.5)/tw, 1-(np.arange(th)+.5)/th)
        generated = origin+xx.reshape(-1, 1)*u+yy.reshape(-1, 1)*v
        points = generated @ inverse[:3, :3].T+inverse[:3, 3]
        region = np.ones((th, tw), np.uint8)
        if polygon is not None:
            region[:] = 0
            polygon_uv = (np.asarray(polygon) - origin[:2]) / np.array([u[0], v[1]])
            contour = np.rint(np.column_stack([polygon_uv[:, 0]*tw-.5, (1-polygon_uv[:, 1])*th-.5])).astype(np.int32)
            cv2.fillPoly(region, [contour], 1)
        normal = np.cross(u, v); normal /= np.linalg.norm(normal)
        if outward is not None and np.dot(normal, outward) < 0:
            normal = -normal
        normal_vg = inverse[:3, :3] @ normal; normal_vg /= np.linalg.norm(normal_vg)
        in_view = np.zeros(len(points), bool)
        depth_supported = np.zeros(len(points), bool)
        after_occlusion = np.zeros(len(points), bool)
        after_semantic = np.zeros(len(points), bool)
        geometric_support = np.zeros(len(points), bool)
        geometric_used = np.zeros(len(points), bool)
        def project(i, extend=False, geometry_evidence=False):
            uv, visible, z = photo_visibility(points, es[i], ks[i], depths[i], confidences[i], extension_tolerance if extend else projection_tolerance)
            if extend and kind in ('floor','ceiling'):
                pixel=np.rint(np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1)).astype(np.int64)
                visible &= horizontal_regions[i,np.clip(pixel[:,1],0,h-1),np.clip(pixel[:,0],0,w-1)]
            if not extend and not geometry_evidence: in_view[:] |= (z > 0) & np.isfinite(uv).all(1) & (uv[:, 0] >= 1) & (uv[:, 0] < w-2) & (uv[:, 1] >= 1) & (uv[:, 1] < h-2)
            if not extend and not geometry_evidence: depth_supported[:] |= visible
            camera_center = -es[i, :, :3].T @ es[i, :, 3]
            blockers = texture_blockers if object_box is None else [box for box in texture_blockers if box['id'] != object_box['id']]
            if blockers:
                indices=np.flatnonzero(visible)
                camera_room=world_to_room[:3,:3]@camera_center+world_to_room[:3,3]
                visible[indices]&=~furniture_occlusion(generated[indices],camera_room,blockers)
            if not extend and not geometry_evidence: after_occlusion[:] |= visible
            labelled = semantic_masks.allowed(i, uv, kind, object_box, wall_margin=12 if extend and kind=='wall' else 0)
            if kind == 'wall':
                _, strict_depth, _ = photo_visibility(points, es[i], ks[i], depths[i], confidences[i], .07/(scale*output_scale))
                geometric = strict_depth & matching_normals(normals[i], uv, normal_vg)
                geometric &= semantic_masks.allowed(i, uv, kind, allow_unlabelled_wall=True)
                if geometry_evidence:
                    visible &= geometric
                else:
                    fallback = visible & geometric & geometric_support & ~labelled
                    if not extend and not geometry_evidence: geometric_used[:] |= fallback
                    visible &= labelled | (geometric & geometric_support)
            else:
                visible &= labelled
            if not extend and not geometry_evidence: after_semantic[:] |= visible
            view = camera_center-points
            facing = (view @ normal_vg)/np.maximum(np.linalg.norm(view, axis=1), 1e-8)
            if outward is None:
                facing = np.abs(facing)
            border = np.minimum.reduce([uv[:, 0], w-uv[:, 0], uv[:, 1], h-uv[:, 1]])
            score = facing**2 / np.maximum(z**2, 1e-8) * np.clip(border/25, 0, 1) * sharpness[i]
            take = visible & (facing > .22) & region.ravel().astype(bool)
            return take, score
        if kind == 'wall':
            geometric_support, geometric_votes = independent_support(viewpoint_groups,len(points),lambda i:project(i,geometry_evidence=True))
        def sample(i, take):
            uv, _, _ = photo_visibility(points[take], es[i], ks[i], depths[i], confidences[i], projection_tolerance)
            photo, photo_scale = photos.get(i)
            return sample_photo(photo, photo_scale, uv) * gains[i]
        texture, supported, frames, secondary, blend_weight, selected, photo_selection = regional_face_texture((th, tw), color, n, project, sample)
        completion=np.zeros(len(points),bool)
        extension_frames=np.full(len(points),-1,np.int32)
        if kind in ('floor','wall','ceiling'):
            texture,completion,extension_frames=extend_same_photos(texture,supported,selected,lambda i:project(i,extend=True),sample)
            Image.fromarray(completion.reshape(th,tw).astype(np.uint8)*255).save(output/f'{name}-photo-extension.png')
            Image.fromarray((extension_frames.reshape(th,tw)+1).astype(np.uint16)).save(output/f'{name}-extension-source.png')
        if kind == 'wall':
            Image.fromarray((geometric_used.reshape(th,tw)*255).astype(np.uint8)).save(output/f'{name}-geometric-wall.png')
        coverage = float(supported.sum() / max(int(region.sum()), 1))
        Image.fromarray(texture.reshape(th, tw, 3)).save(output/f'{name}.jpg', quality=95, subsampling=0)
        Image.fromarray((supported.reshape(th, tw)*255).astype(np.uint8)).save(output/f'{name}-support.png')
        # Index 0 is unobserved; other values are manifest frame indices + 1.
        Image.fromarray((frames.reshape(th, tw)+1).astype(np.uint16)).save(output/f'{name}-source.png')
        Image.fromarray((secondary.reshape(th, tw)+1).astype(np.uint16)).save(output/f'{name}-secondary.png')
        Image.fromarray(np.rint(blend_weight.reshape(th, tw)*255).astype(np.uint8)).save(output/f'{name}-blend.png')
        area = float(np.linalg.norm(np.cross(u, v)))
        if polygon is not None:
            xy = np.asarray(polygon)
            area = abs(float(np.sum(xy[:, 0]*np.roll(xy[:, 1], -1)-xy[:, 1]*np.roll(xy[:, 0], -1)))) / 2
        used_frames = np.unique(np.concatenate([frames[supported], secondary[secondary >= 0]])).tolist()
        surfaces.append(dict(id=name, origin=origin.tolist(), u=u.tolist(), v=v.tolist(), kind=kind,
                             object_id=object_box['id'] if object_box else None,
                             object_label=object_box['label'] if object_box else None,
                             texture=f'{name}.jpg', support=f'{name}-support.png', polygon=polygon,
                             photo_provenance=dict(dominant=f'{name}-source.png', secondary=f'{name}-secondary.png',
                                                   secondary_weight=f'{name}-blend.png', frame_index_offset=1),
                             photo_selection=photo_selection, semantic_support_mode=semantic_masks.mode(kind, object_box),
                             surface_area=area*output_scale**2, supported_fraction=coverage, frames_used=used_frames,
                             blended_fraction=float(np.count_nonzero(blend_weight)/max(int(supported.sum()), 1)),
                             projection_diagnostics={key: float(np.count_nonzero(mask & region.ravel().astype(bool)) / max(int(region.sum()), 1))
                                 for key, mask in [('in_view_fraction', in_view), ('depth_supported_fraction', depth_supported),
                                                   ('after_box_occlusion_fraction', after_occlusion), ('after_furniture_mask_fraction', after_semantic),
                                                   ('after_semantic_mask_fraction', after_semantic)]},
                             photo_extension=dict(method='same_source_photo_projection',filled_fraction=float(completion.sum()/max(int(region.sum()),1)),mask=f'{name}-photo-extension.png' if kind in ('floor','wall','ceiling') else None,source=f'{name}-extension-source.png' if kind in ('floor','wall','ceiling') else None,source_frame_index_offset=1,position_verified=False),
                             geometric_wall_support=dict(enabled=kind=='wall',minimum_viewpoints=3,normal_cosine=.97,depth_tolerance_model_units=.07,mask=f'{name}-geometric-wall.png' if kind=='wall' else None,additional_candidate_fraction=float(geometric_used.sum()/max(int(region.sum()),1))),
                             texture_size=[tw, th]))
        if polygon is not None:
            xy = np.asarray(polygon)
            vertices = np.column_stack([xy,np.full(len(xy),origin[2])])
            uv_coords = (xy-origin[:2])/np.array([u[0],v[1]])
            faces = triangulate_polygon(xy)
        else:
            vertices = np.array([origin,origin+u,origin+u+v,origin+v])
            uv_coords = np.array([[0,0],[1,0],[1,1],[0,1]])
            faces = np.array([[0,1,2],[0,2,3]])
        if kind=='floor' or (kind=='wall' and np.sum(footprint[:,0]*np.roll(footprint[:,1],-1)-footprint[:,1]*np.roll(footprint[:,0],-1))<0):
            faces=faces[:,::-1]
        if outward is not None and np.dot(np.cross(u,v),outward)<0:
            faces=faces[:,::-1]
        visual = trimesh.visual.texture.TextureVisuals(uv=uv_coords, material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(texture.reshape(th,tw,3)), metallicFactor=0,
            roughnessFactor=1,doubleSided=True))
        exported.add_geometry(trimesh.Trimesh(vertices,faces,visual=visual,process=False),node_name=name,geom_name=name)
        print(f'{name}: photo coverage {coverage:.0%}', flush=True)

    lo, hi = footprint.min(0), footprint.max(0)
    for kind, z, color in [('floor', floor, [86, 85, 81]), ('ceiling', ceiling, [211, 210, 203])]:
        surface(kind, [*lo, z], [hi[0]-lo[0], 0, 0], [0, hi[1]-lo[1], 0], color, kind, footprint.tolist())
    # Reuse exactly the same perimeter vertices as the floor/ceiling; no wall seams.
    for i, (a,b) in enumerate(zip(footprint,np.roll(footprint,-1,axis=0))):
        surface(f'wall-{i}', [*a,floor], [*(b-a),0], [0,0,ceiling-floor], [208,205,187], 'wall')
    for b in boxes:
        c, r, s = np.asarray(b['center']), np.asarray(b['rotation']), np.asarray(b['scale'])
        if b['class']=='bbox':
            parts = furniture_parts(b, floor)
            if not parts:continue
            models.append(dict(id=b['id'],label=b['label'],parts=parts,
                               storage_geometry=b.get('evidence',{}).get('storage_geometry')))
            if b['label'] not in ('curtain','illumination','chandelier','painting','wall_decoration'):
                colliders.append(dict(id=b['id'],center=b['center'],rotation=b['rotation'],scale=b['scale'],
                                      bottom=min(p['center'][2]-p['scale'][2]/2 for p in parts),
                                      top=max(p['center'][2]+p['scale'][2]/2 for p in parts)))
            for part in parts:
                pc, pr, ps = (np.asarray(part[key]) for key in ('center','rotation','scale'))
                for axis in range(3):
                    a, d = [j for j in range(3) if j != axis]
                    for sign in (-1, 1):
                        u,v = pr[:,a]*ps[a],pr[:,d]*ps[d]
                        surface(f'object-{b["id"]}-{part["name"]}-{axis}-{sign}', pc+sign*pr[:,axis]*ps[axis]/2-u/2-v/2,
                                u,v,[181,178,158],'object',object_box=b,outward=sign*pr[:,axis])
    actual_shell=trimesh.util.concatenate([mesh for name,mesh in exported.geometry.items() if name in ('floor','ceiling') or name.startswith('wall-')])
    actual_shell.merge_vertices(merge_tex=True,merge_norm=True)
    if not actual_shell.is_watertight or not actual_shell.is_winding_consistent:
        raise ValueError('Textured room mesh failed closed-shell validation')
    glb_from_room = trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0])
    exported.apply_scale(output_scale)
    exported.apply_transform(glb_from_room)
    glb = exported.export(file_type='glb')
    (output/'room.glb').write_bytes(glb)
    # Spawn is recomputed after scale calibration using a physical body radius.
    spawn = [0., 0., 0.]
    report = dict(schema_version=2, method='Closed room shell + simple furniture solids + depth-checked original photos',
                  model_url='room.glb', glb_from_room=glb_from_room.tolist(), models=models,
                  geometry_validation=dict(shell_watertight=True,shell_volume=float(shell.volume),
                                           shell_vertices=len(shell_vertices),shell_triangles=len(shell_faces)),
                  navigation=dict(footprint=footprint.tolist(),floor=floor,ceiling=ceiling,colliders=colliders,
                                  radius=.18,eye_height=min(1.55,(ceiling-floor)*.65),doors='closed',spawn=spawn),
                  appearance_generation_model=None, source_prediction=prediction,
                  image_furniture_summary=({'confirmed_count':layout['image_furniture'].get('confirmed_count',layout.get('counts',{}).get('bboxes',0))} if 'image_furniture' in layout else None),
                  storage_geometry_summary={key:layout.get('storage_geometry',{}).get(key) for key in
                      ('version','status','accepted_count','preserved_count','wall_correction')},
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_npz_sha256=transform['source_npz_sha256'], transform=transform,
                  metric_scale_verified=False, vertical_anchors=layout['vertical_anchors'], floor=floor, ceiling=ceiling, footprint=footprint.tolist(),
                  surfaces=surfaces, furniture=[],
                  caveats=['Furniture detail is procedural, not recovered geometry.', 'Unseen texels use a plain material.',
                           'Estimated furniture boxes mask projected photos on architecture to avoid baking furniture into walls.',
                           'Detected furniture and unknown image regions cannot texture architecture, independently of the estimated boxes.',
                           'Unlabelled wall pixels require three camera-origin groups, matching depth and normals, and furniture/unknown exclusion; legacy layouts retain depth-only fallback.',
                           'Depth tolerance is 0.14 assumed metres; foreground rejection is stricter.',
                           'Doors and windows are closed surfaces; adjacent unobserved rooms are not generated.',
                           'Unobserved furniture supports are simplified inferred parts.',
                           'This is not SpatialGen output.'])
    report = scale_room_report(report, output_scale)
    report['navigation']['eye_height'] = min(1.55, (report['ceiling']-report['floor'])*.65)
    report['navigation']['spawn'] = navigation_spawn(report['footprint'], report['navigation']['colliders'],
        report['floor'], report['navigation']['eye_height'], report['navigation']['radius'])
    report['calibration'] = calibration
    report['metric_scale_calibrated'] = calibration['metric_scale_calibrated']
    if calibration['metric_scale_calibrated']:
        report['transform']['unit'] = ('metres from RGB-D sensor calibration; local shape is unverified' if
            calibration.get('source') == 'sensor_calibration' else 'metres calibrated from user measurement; local shape is unverified')
    provenance_keys = ('name', 'source_id', 'source_type', 'source_kind', 'kind', 'source_name', 'source_frame_index',
                       'timestamp_seconds', 'timestamp', 'panorama_face', 'panorama_projection', 'panorama_group',
                       'face_camera_to_panorama', 'capture_id', 'asset_id', 'provenance')
    photo_sources = [dict(index=i, **{key: record[key] for key in provenance_keys if key in record},
                          color_gain=gains[i].tolist(), texture_scale=photos.scales.get(i, 1)) for i, record in enumerate(records)]
    quality = capture_quality_report(report['surfaces'], calibration, photo_sources)
    quality['semantic_textures'] = semantic_masks.summary()
    quality['metrics']['max_photos_per_face'] = 6
    quality['metrics']['legacy_texture_fallback_surface_count'] = sum(s['semantic_support_mode'].startswith('legacy_') for s in surfaces)
    for item, surface in zip(quality['surfaces'], report['surfaces']):
        item.update(semantic_support_mode=surface['semantic_support_mode'], photo_selection=surface['photo_selection'])
    by_surface = {surface['id']: surface for surface in report['surfaces']}
    for recommendation in quality['recommendations']:
        surface = by_surface.get(recommendation.get('surface_id'))
        if surface is None or recommendation.get('code') != 'furniture_mask_occlusion':
            continue
        diagnostics = surface['projection_diagnostics']
        if diagnostics['after_box_occlusion_fraction'] - diagnostics['after_semantic_mask_fraction'] > .2:
            recommendation.update(code='semantic_photo_support_missing',
                message=('壁と判断できる画像領域が不足しています。家具や不明な領域を避け、根拠が足りない部分は無地で残しています。' if
                         surface['semantic_support_mode'] == 'positive_wall_mask' else
                         '対象と確認できる画像領域が不足しています。別の家具や不明な領域を避け、根拠が足りない部分は無地で残しています。'))
    if semantic_masks.errors:
        quality['recommendations'].insert(0, dict(id='semantic-mask-unavailable', code='semantic_mask_unavailable',
            kind='appearance', severity='high', message='画像の領域情報を読み込めないため、一部の面は無地で残しています。生成結果を作り直してください。'))
        quality['summary']['needs_review'] = True
    if layout['vertical_anchors'].get('ceiling_observed') is False:
        quality['recommendations'].insert(0, dict(id='ceiling-unobserved', code='unobserved_structure',
            kind='ceiling', severity='high', surface_id='ceiling', object_id=None,
            message='天井の形は仮の推定です。壁とのつなぎ目を含めて天井を撮影してください。'))
        quality['summary']['needs_review'] = True
    if alignment:
        report['photo_alignment'] = dict(report_url=f'../{prediction}/alignment/report.json',
            accepted_planes=alignment['geometry']['accepted_count'],accepted_cameras=alignment['cameras']['accepted_count'])
    report['photo_extension'] = dict(enabled=True,kinds=['floor','wall','ceiling'],method='same_source_photo_projection',strict_support_unchanged=True,position_verified=False,max_depth_tolerance_model_units=.35/output_scale)
    report['quality_url'] = 'quality.json'
    report['quality_summary'] = quality['summary']
    report['appearance_processing'] = dict(method='regional_quality_photos_v3', max_photos_per_face=n,
        max_photos_per_object_face=n,
        max_photo_cache_bytes=photos.max_bytes, color_gain_limit=[.9, 1.1], photo_sources=photo_sources,
        semantic_masks=semantic_masks.summary(), provenance_frame_index_offset=1)
    report['caveats'].extend(['Exposure correction uses only shared depth-supported samples and is limited to 10%.',
                             'All frames are considered per surface. Locally smoothed visibility-verified quality selects one photo per texel without cross-view color averaging.',
                             'Measurements calibrate uniform scale and do not independently verify geometry.'])
    (output/'quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'room.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return complete_materials(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='spatiallm-001')
    parser.add_argument('--prediction', default='prediction-reviewed-002')
    parser.add_argument('--output', default='solid-photo-001')
    parser.add_argument('--options', type=Path, help='Optional capture/measurement options JSON')
    args = parser.parse_args()
    for value in (args.run, args.prediction, args.output):
        if not value.replace('-', '').isalnum():
            raise ValueError('Invalid identifier')
    options = json.loads(args.options.read_text(encoding='utf-8')) if args.options else None
    build(ROOT/'data/generation-runs'/args.run, args.prediction,
          ROOT/'web/generation-reviews'/args.run/args.output, options=options)
