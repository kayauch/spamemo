"""Synthetic scene tests use fixed instance masks and independent camera origins."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

from spatial_memory.storage_geometry import refine_storage_layout
from spatial_memory.storage_visibility import mask_metrics
from spatial_memory.storage_side_outline import trace_side_outline


def camera_at(center,target):
    center=np.asarray(center,float);target=np.asarray(target,float)
    forward=(target-center)/np.linalg.norm(target-center)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    down=np.cross(forward,right);r=np.stack([right,down,forward])
    return np.column_stack((r,-r@center))


def synthesize(es,k,thickness=.4):
    # Axis-aligned analytic box, independent of the candidate-oriented renderer.
    yy,xx=np.indices((360,480));rays=np.stack([xx,yy,np.ones_like(xx)],axis=-1)@np.linalg.inv(k).T@es[:,:3]
    center=-es[:,:3].T@es[:,3];low=np.array([-thickness,-1.,0.]);high=np.array([0.,1.,2.])
    with np.errstate(divide='ignore',invalid='ignore'):
        first=(low-center)/rays;last=(high-center)/rays
    near=np.minimum(first,last);far=np.maximum(first,last)
    enter=near.max(axis=-1);leave=far.min(axis=-1)
    mask=(leave>=np.maximum(enter,0))&(leave>0)
    # Only front depths are reliable: the side must be recovered from outlines.
    front=mask&(near.argmax(axis=-1)==0)
    return np.where(mask,enter,10.),np.where(front,4.,1.),mask


class StorageGeometryTests(unittest.TestCase):
    def make_scene(self,root,centers=None):
        if centers is None:centers=[[4.,3.,1.5],[3.,3.7,1.5],[3.7,4.5,1.5]]
        es=np.array([camera_at(c,[-.2,0,1.]) for c in centers]);k=np.array([[440.,0,240.],[0,440.,180.],[0,0,1.]])
        depths=[];conf=[];detections=[];ids=[]
        for i,e in enumerate(es):
            d,c,m=synthesize(e,k);depths.append(d);conf.append(c)
            identifier=f'detection-{i}';ids.append(identifier);path=f'{identifier}.png'
            Image.fromarray(m.astype('uint8')*255).save(root/path)
            detections.append(dict(frame=i,detection_id=identifier,label='cabinet',kind='furniture',mask=path,viewpoint_id=f'view-{i}'))
        box=dict(id=73,label='cabinet',**{'class':'bbox'},center=[0.,0.,1.],scale=[.06,2.,2.],rotation=np.eye(3).tolist(),
                 source_detection_ids=ids,source_frames=list(range(len(es))),evidence=dict(valid=True),
                 vg_world=dict(center=[0.,0.,1.],scale=[.06,2.,2.],rotation=np.eye(3).tolist()))
        wall=dict(id=1,**{'class':'wall'},center=[-.9,0,1],scale=[.02,3,2.5],rotation=np.eye(3).tolist(),evidence=dict(valid=True))
        layout=dict(boxes=[wall,box],image_furniture=dict(detections=detections,quality_report=dict(objects=[dict(id=73,dimensions_model_units=box['scale'],center_model_units=box['center'])])))
        records=[dict(index=i,kind='video',source_id='capture') for i in range(len(es))]
        return layout,np.array(depths),np.array(conf),es,np.repeat(k[None],len(es),axis=0),records

    def test_recovers_side_without_reliable_side_depth_and_keeps_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);layout,d,c,e,k,records=self.make_scene(root)
            before=deepcopy(layout)
            out=refine_storage_layout(layout,root,d,c,e,k,np.eye(4),records)
            report=out['image_furniture']['storage_geometry']['objects'][0]
            self.assertTrue(report['accepted'],report)
            self.assertAlmostEqual(report['depth_estimate'],.4,delta=.045)
            self.assertEqual(out['boxes'][1]['id'],73)
            self.assertEqual(out['boxes'][1]['source_detection_ids'],layout['boxes'][1]['source_detection_ids'])
            self.assertEqual(out['boxes'][0],layout['boxes'][0])
            self.assertEqual(layout,before)
            self.assertEqual(out['boxes'][1]['vg_world']['center'],out['boxes'][1]['center'])
            self.assertFalse(report['complete_hidden_geometry_verified'])
            quality=out['image_furniture']['quality_report']['objects'][0]
            self.assertEqual(quality['dimensions_model_units'],out['boxes'][1]['scale'])
            self.assertFalse(quality['hidden_surfaces_verified'])

    def test_same_camera_origins_are_not_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);layout,d,c,e,k,records=self.make_scene(root,[[4,3,1.5]]*3)
            out=refine_storage_layout(layout,root,d,c,e,k,np.eye(4),records)
            report=out['image_furniture']['storage_geometry']['objects'][0]
            self.assertFalse(report['accepted']);self.assertIn('fewer_than_three_independent_viewpoints',report['reasons'])
            self.assertEqual(out['boxes'][1]['scale'],layout['boxes'][1]['scale'])

    def test_held_mask_is_fixed_not_refitted_or_replaced_by_other_storage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);layout,d,c,e,k,records=self.make_scene(root)
            original=np.asarray(Image.open(root/'detection-2.png'))
            # Another instance has the correct silhouette. The target's fixed
            # held silhouette differs; using all-storage masks would hide this.
            Image.fromarray(original).save(root/'other.png')
            shifted=np.roll(original,45,axis=1);Image.fromarray(shifted).save(root/'detection-2.png')
            layout['image_furniture']['detections'].append(dict(frame=2,detection_id='other',label='cabinet',kind='furniture',mask='other.png'))
            out=refine_storage_layout(layout,root,d,c,e,k,np.eye(4),records)
            report=out['image_furniture']['storage_geometry']['objects'][0]
            self.assertFalse(report['accepted']);self.assertEqual(out['boxes'][1]['scale'],layout['boxes'][1]['scale'])

    def test_target_depth_error_is_not_treated_as_foreground_occlusion(self):
        shape=(15,15);target=np.ones(shape,bool)
        metrics,_=mask_metrics(np.full(shape,2.),np.ones(shape),np.full(shape,3.),target,target,np.zeros(shape,bool))
        self.assertEqual(metrics['known_occluded_pixels'],0)
        self.assertEqual(metrics['target_depth_contradiction_fraction'],1.)

    def test_front_only_view_preserves_original_and_wall(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);layout,d,c,e,k,records=self.make_scene(root,[[3.,0.,1.5],[4.,0.,1.5],[5.,0.,1.5]])
            out=refine_storage_layout(layout,root,d,c,e,k,np.eye(4),records)
            report=out['image_furniture']['storage_geometry']['objects'][0]
            self.assertFalse(report['accepted']);self.assertIn('insufficient_independent_side_outline_support',report['reasons'])
            for field in ['center','rotation','scale','vg_world','source_detection_ids']:
                self.assertEqual(out['boxes'][1][field],layout['boxes'][1][field])
            self.assertEqual(out['boxes'][0],layout['boxes'][0])
            quality=out['image_furniture']['quality_report']['objects'][0]
            self.assertIn('storage_side_depth_unresolved',[r['code'] for r in quality['needs_photo']])

    def test_generation_transform_keeps_vg_world_consistent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);layout,d,c,e,k,records=self.make_scene(root)
            theta=.4;r=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
            tr=np.eye(4);tr[:3,:3]=r*1.7;tr[:3,3]=[2.,-1.,.5]
            original=layout['boxes'][1]
            original['center']=(tr[:3,:3]@original['center']+tr[:3,3]).tolist()
            original['scale']=(np.array(original['scale'])*1.7).tolist();original['rotation']=r.tolist()
            out=refine_storage_layout(layout,root,d,c,e,k,dict(vg_world_to_generation_world=tr.tolist(),generation_world_to_vg_world=np.linalg.inv(tr).tolist()),records)
            report=out['storage_geometry']['objects'][0]
            self.assertTrue(report['accepted'],report)
            b=out['boxes'][1];vg=b['vg_world']
            np.testing.assert_allclose(tr[:3,:3]@vg['center']+tr[:3,3],b['center'],atol=1e-8)
            np.testing.assert_allclose(r@vg['rotation'],b['rotation'],atol=1e-8)
            np.testing.assert_allclose(np.array(vg['scale'])*1.7,b['scale'],atol=1e-8)

    def test_unknown_or_conflicting_full_overlay_never_accepts(self):
        for kind,label in [('unknown','unknown'),('furniture','cabinet'),('wall','wall')]:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);layout,d,c,e,k,records=self.make_scene(root)
                for f in range(3):
                    layout['image_furniture']['detections'].append(dict(frame=f,detection_id=f'conflict-{f}',
                        kind=kind,label=label,mask=f'detection-{f}.png'))
                out=refine_storage_layout(layout,root,d,c,e,k,np.eye(4),records)
                report=out['storage_geometry']['objects'][0]
                self.assertFalse(report['accepted']);self.assertIn('ambiguous_instance_mask_ownership',report['reasons'])
                self.assertEqual(out['boxes'][1]['scale'],layout['boxes'][1]['scale'])

    def test_tiny_boundary_conflict_removed_before_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);layout,d,c,e,k,records=self.make_scene(root)
            for f in range(3):
                mask=np.asarray(Image.open(root/f'detection-{f}.png'))>0
                y,x=np.argwhere(mask)[0];ambiguous=np.zeros_like(mask);ambiguous[y,x]=True
                Image.fromarray(ambiguous.astype('uint8')*255).save(root/f'unknown-{f}.png')
                layout['image_furniture']['detections'].append(dict(frame=f,detection_id=f'unknown-{f}',kind='unknown',label='unknown',mask=f'unknown-{f}.png'))
            out=refine_storage_layout(layout,root,d,c,e,k,np.eye(4),records)
            report=out['storage_geometry']['objects'][0]
            self.assertTrue(report['accepted'],report)
            for row in report['mask_ownership']:
                self.assertGreater(row['removed_boundary_pixels'],0)
                self.assertLess(row['fixed_target_pixels'],row['target_pixels'])

    def test_removed_unknown_strip_cannot_become_rear_edge(self):
        es=camera_at([4.,3.,1.5],[-.2,0,1.]);k=np.array([[440.,0,240.],[0,440.,180.],[0,0,1.]])
        _,_,full=synthesize(es,k,.4);_,_,fragment=synthesize(es,k,.2)
        candidate=dict(front_normal=[1.,0.],front=0.,tangent_range=[-1.,1.],height_range=[0.,2.])
        unsafe=trace_side_outline(candidate,es,k,np.eye(4),fragment,max_depth=.8)
        self.assertIsNotNone(unsafe)
        self.assertAlmostEqual(unsafe['depth_median'],.2,delta=.03)
        guarded=trace_side_outline(candidate,es,k,np.eye(4),fragment,forbidden_mask=full&~fragment,max_depth=.8)
        self.assertIsNone(guarded)

    def test_outlying_training_depth_is_reported_and_rejected_after_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);layout,d,c,e,k,records=self.make_scene(root,[[4.,3.,1.5],[3.,3.7,1.5],[3.7,4.5,1.5],[4.5,3.5,1.5]])
            d[0]*=1.3
            out=refine_storage_layout(layout,root,d,c,e,k,np.eye(4),records)
            report=out['storage_geometry']['objects'][0]
            self.assertFalse(report['accepted'],report)
            self.assertIn('source_instance_geometry_contradiction',report['reasons'])
            by_frame={m['frame']:m for m in report['validation']}
            self.assertEqual(set(by_frame),{0,1,2,3})
            self.assertFalse(by_frame[0]['held']);self.assertTrue(by_frame[2]['held'])
            self.assertGreater(by_frame[0]['target_depth_contradiction_fraction'],.12)
            self.assertIn(0,report['failed_validation_frames'])
            self.assertEqual(out['boxes'][1]['scale'],layout['boxes'][1]['scale'])


if __name__=='__main__':unittest.main()
