"""Reject the actual failure mode: low pixel error but fragmented reconstruction."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('prepare_video_colmap',
    Path(__file__).parents[1]/'scripts/prepare_video_colmap.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('fraction', [15/56, 32/72])
def test_partial_models_are_not_accepted_as_whole_rooms(fraction):
    with pytest.raises(RuntimeError, match='coverage'):
        module.validate_reconstruction(dict(registered_fraction=fraction, points=1100,
                                            mean_reprojection_error=.85))


@pytest.mark.parametrize('error', [float('nan'), float('inf'), -1., 3.])
def test_invalid_camera_errors_are_rejected(error):
    with pytest.raises(RuntimeError, match='error'):
        module.validate_reconstruction(dict(registered_fraction=1., points=1100,
                                            mean_reprojection_error=error))


def test_connected_low_error_model_is_accepted():
    module.validate_reconstruction(dict(registered_fraction=.95, points=1100,
                                        mean_reprojection_error=.85))
