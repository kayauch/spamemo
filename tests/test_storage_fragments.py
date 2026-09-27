"""Independent complementary observations, never arbitrary depth inflation."""
from copy import deepcopy
import unittest
import numpy as np

from spatial_memory.storage_fragments import consolidate_storage_fragments


def box(identifier, center, size, frames, label='cabinet', **extra):
    return dict(id=identifier, **{'class': 'bbox'}, label=label, center=center,
                scale=size, rotation=np.eye(3).tolist(), source_frames=frames,
                source_detection_ids=[f'{f}-{identifier}' for f in frames],
                detection_scores=[.8]*len(frames), evidence=dict(valid=True, points_inside=100,
                    quality=dict(viewpoint_ids=[f'v{f}' for f in frames], labels_observed=[label])), **extra)


def scene():
    # A front covers y=-1..1; an independent side has its front at x=0.
    front = box(1, [0., 0., 1.], [.06, 2., 2.], [0, 1])
    side = box(2, [-.2, -.65, 1.], [.4, .6, 2.], [2, 3])
    return dict(boxes=[front, side], image_furniture=dict(confirmed_count=2,
        quality_report=dict(objects=[dict(id=1), dict(id=2)])))


class StorageFragmentTests(unittest.TestCase):
    def test_observed_union_preserves_sources_id_and_input(self):
        original = scene(); before = deepcopy(original)
        result = consolidate_storage_fragments(original, np.eye(4))
        front, side = result['boxes']
        self.assertEqual(original, before)
        self.assertEqual(front['id'], 1)
        np.testing.assert_allclose(front['scale'], [.43, 2., 2.])
        np.testing.assert_allclose(front['center'], [-.185, 0., 1.])
        self.assertEqual(front['source_frames'], [0, 1, 2, 3])
        self.assertEqual(len(front['source_detection_ids']), 4)
        self.assertTrue(side['hidden']); self.assertEqual(side['merged_into'], 1)
        self.assertEqual(side['source_detection_ids'], before['boxes'][1]['source_detection_ids'])
        report = front['evidence']['storage_fragment_consolidation']
        self.assertFalse(report['complete_hidden_geometry_verified'])
        self.assertTrue(report['hidden_closures_inferred'])
        self.assertEqual(result['image_furniture']['confirmed_count'], 1)
        summaries = result['image_furniture']['quality_report']['objects']
        self.assertEqual(summaries[0]['dimensions_model_units'], front['scale'])
        self.assertEqual(summaries[0]['frame_count'], 4)
        self.assertTrue(summaries[1]['hidden'])
        self.assertEqual(consolidate_storage_fragments(result)['boxes'], result['boxes'])

    def test_overlapping_partial_fronts_share_one_observed_side(self):
        data = scene()
        data['boxes'].append(box(3, [.015, .5, 1.], [.06, 1.4, 2.], [4, 5]))
        out = consolidate_storage_fragments(data)
        self.assertEqual(out['storage_fragment_consolidation']['merged_count'], 1)
        self.assertEqual(out['boxes'][2]['merged_into'], 1)
        self.assertEqual(out['boxes'][0]['source_frames'], [0, 1, 2, 3, 4, 5])

    def test_same_frame_instances_never_merge(self):
        data = scene(); data['boxes'][1]['source_frames'] = [1, 2]
        out = consolidate_storage_fragments(data)
        self.assertEqual(out['boxes'], data['boxes'])

    def test_front_members_do_not_share_source_frames(self):
        data = scene(); data['boxes'].append(box(3, [0., .2, 1.], [.06, 1.5, 2.], [1, 5]))
        out = consolidate_storage_fragments(data)
        self.assertNotIn('merged_into', out['boxes'][2])

    def test_unrelated_rows_and_non_storage_untouched(self):
        for changes in [dict(center=[-.2, 3., 1.]), dict(center=[-.6, -.65, 1.]),
                        dict(center=[-.2, -.65, 4.]), dict(label='chair'), dict(label='box'),
                        dict(rotation=[[0,-1,0],[1,0,0],[0,0,1]])]:
            with self.subTest(changes=changes):
                data = scene(); data['boxes'][1].update(changes)
                self.assertEqual(consolidate_storage_fragments(data)['boxes'], data['boxes'])

    def test_side_on_both_sides_is_ambiguous(self):
        data = scene(); data['boxes'].append(box(3, [.2, .65, 1.], [.4, .6, 2.], [4, 5]))
        self.assertEqual(consolidate_storage_fragments(data)['boxes'], data['boxes'])

    def test_incompatible_rear_bounds_are_not_averaged(self):
        data = scene(); data['boxes'].append(box(3, [-.4, .4, 1.], [.8, 1.2, 2.], [4, 5]))
        self.assertEqual(consolidate_storage_fragments(data)['boxes'], data['boxes'])

    def test_no_depth_added_without_observed_side(self):
        data = scene(); data['boxes'][1]['scale'][0] = .06
        data['boxes'][1]['center'][0] = 0.
        self.assertEqual(consolidate_storage_fragments(data)['boxes'], data['boxes'])

    def test_already_solid_or_verified_objects_unchanged(self):
        data = scene(); data['boxes'][0]['scale'][0] = .4
        self.assertEqual(consolidate_storage_fragments(data)['boxes'], data['boxes'])

    def test_user_corrections_and_invalid_evidence_preserved(self):
        for field in ('user_corrected', 'invalid_evidence', 'user_correction'):
            data = scene()
            if field == 'invalid_evidence': data['boxes'][0]['evidence']['valid'] = False
            elif field == 'user_correction': data['boxes'][0]['user_correction'] = dict(scale=[.06,2,2])
            else: data['boxes'][0]['evidence']['user_corrected'] = True
            self.assertEqual(consolidate_storage_fragments(data)['boxes'], data['boxes'])

    def test_old_validation_is_superseded_and_retained(self):
        data = scene()
        prior = dict(id=1, status='preserved_original', accepted=False, reasons=['insufficient_side_views'])
        data['boxes'][0]['evidence']['storage_geometry'] = deepcopy(prior)
        data['storage_geometry'] = dict(objects=[deepcopy(prior)], accepted_count=0, preserved_count=1)
        out = consolidate_storage_fragments(data)
        report = out['boxes'][0]['evidence']['storage_geometry']
        self.assertEqual(report['prior_validation'], prior)
        self.assertFalse(report['current_geometry_validated'])
        self.assertEqual(out['storage_geometry']['objects'][0], report)
        data = scene(); data['boxes'][0]['evidence']['storage_geometry'] = dict(accepted=True)
        self.assertEqual(consolidate_storage_fragments(data)['boxes'], data['boxes'])

    def test_coordinate_transform_updates_saved_world_consistently(self):
        data = scene(); theta=.4
        tr=np.eye(4); tr[:3,:3]=np.array([[np.cos(theta),-np.sin(theta),0],
            [np.sin(theta),np.cos(theta),0],[0,0,1]])*2; tr[:3,3]=[4,5,6]
        out=consolidate_storage_fragments(data,dict(vg_world_to_generation_world=tr.tolist()))
        b=out['boxes'][0]; vg=b['vg_world']
        np.testing.assert_allclose(tr[:3,:3]@vg['center']+tr[:3,3], b['center'], atol=1e-8)
        np.testing.assert_allclose(np.array(vg['scale'])*2, b['scale'])
        np.testing.assert_allclose(tr[:3,:3]@vg['rotation']/2, np.eye(3),atol=1e-8)


if __name__ == '__main__':
    unittest.main()
