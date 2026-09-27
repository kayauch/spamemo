"""Independent video -> COLMAP baseline. Never reads VGGT or overwrites a run."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pycolmap


def validate_reconstruction(report):
    if report['registered_fraction'] < .8 or report['points'] < 100:
        raise RuntimeError('Insufficient reconstruction coverage; refusing to train a partial room')
    error = report['mean_reprojection_error']
    if not np.isfinite(error) or not 0 <= error <= 2:
        raise RuntimeError('Reprojection error exceeds 2 pixels; refusing unstable cameras')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--max-size', type=int, default=1280)
    args = parser.parse_args()
    if not args.video.is_file() or args.frames < 10 or args.max_size < 320:
        parser.error('Existing video, >=10 frames and >=320 image size required')
    args.output.mkdir(parents=True, exist_ok=False)
    images = args.output / 'input'
    images.mkdir()
    started = time.perf_counter()
    report = {'source': str(args.video.resolve()),
              'source_sha256': hashlib.sha256(args.video.read_bytes()).hexdigest(),
              'geometry_source': 'COLMAP SIFT + bundle adjustment; no VGGT',
              'pycolmap_version': pycolmap.__version__, 'status': 'running'}
    def save():
        report['seconds'] = round(time.perf_counter() - started, 2)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    try:
        cap = cv2.VideoCapture(str(args.video))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or count < 10:
            raise ValueError('Video cannot provide enough frames')
        selected = np.unique(np.linspace(0, count - 1, min(count, args.frames)).astype(int))
        report['frames'] = []
        for i, frame in enumerate(selected):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode frame {frame}')
            scale = min(1., args.max_size / max(image.shape[:2]))
            image = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
            name = f'{i:05d}.jpg'
            if not cv2.imwrite(str(images / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise IOError(name)
            report['frames'].append({'name': name, 'source_frame': int(frame)})
        cap.release()
        save()
        database = args.output / 'database.db'
        extraction = pycolmap.FeatureExtractionOptions()
        extraction.num_threads = 8
        pycolmap.extract_features(database, images, camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=pycolmap.ImageReaderOptions(camera_model='SIMPLE_RADIAL'),
            extraction_options=extraction, device=pycolmap.Device.cpu)
        matching = pycolmap.FeatureMatchingOptions()
        matching.num_threads = 8
        pycolmap.match_exhaustive(database, matching_options=matching, device=pycolmap.Device.cpu)
        options = pycolmap.IncrementalPipelineOptions(num_threads=8, random_seed=42,
                                                     max_runtime_seconds=600)
        models = pycolmap.incremental_mapping(database, images, args.output / 'raw', options=options)
        if not models:
            raise RuntimeError('COLMAP could not reconstruct a model; capture more translation/texture')
        best_id, model = max(models.items(), key=lambda pair: pair[1].num_reg_images())
        report.update(registered_images=model.num_reg_images(), points=model.num_points3D(),
                      mean_reprojection_error=model.compute_mean_reprojection_error(),
                      models=len(models), registered_fraction=model.num_reg_images()/len(selected))
        validate_reconstruction(report)
        target = args.output / 'colmap'
        pycolmap.undistort_images(target, args.output/'raw'/str(best_id), images, num_threads=8)
        # GS readers expect sparse/0, whereas the undistorter writes sparse directly.
        sparse = target / 'sparse'
        destination = sparse / '0'
        destination.mkdir()
        for path in list(sparse.iterdir()):
            if path.is_file():
                path.rename(destination/path.name)
        report['status'] = 'ready'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
