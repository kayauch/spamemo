import test from 'node:test';
import assert from 'node:assert/strict';
import {captureTargets,targetPhotoUrl} from '../web/capture-targets.mjs';
import {revisionBody} from '../web/photo-revision.mjs';
test('candidates distinguish photo evidence from geometry errors and collapse furniture faces',()=>{
  const room={surfaces:[{id:'floor',kind:'floor'},{id:'wall-1',kind:'wall'},{id:'wall-2',kind:'wall'},{id:'a',object_id:7,object_label:'shelf'},{id:'b',object_id:7,object_label:'shelf'}],furniture:[{id:7,label:'shelf'},{id:8,label:'chair',hidden:true}]};
  const quality={recommendations:[{surface_id:'floor',code:'projection_depth_mismatch'},{surface_id:'wall-1',code:'missing_photo_coverage',missing_fraction:.9},{surface_id:'wall-2',code:'semantic_photo_support_missing'},{surface_id:'wall-2',code:'projection_depth_mismatch'},{object_id:7,code:'missing_photo_coverage'}],surfaces:[{surface_id:'a',frames_used:[2,3]},{surface_id:'b',frames_used:[3,4]}]};
  const targets=captureTargets(room,quality);assert.equal(targets.length,4);
  assert.deepEqual(targets.filter(t=>t.candidate).map(t=>t.key),['surface:wall-1','object:7']);
  assert.equal(targets[0].repair,true);assert.equal(targets[2].repair,true);
  assert.deepEqual(targets[3].surface_ids,['a','b']);assert.deepEqual(targets[3].frames,[2,3,4]);
});
test('absent quality supports manual selection without inventing a missing-photo diagnosis',()=>{
  const [target]=captureTargets({surfaces:[{id:'floor',kind:'floor'}]});assert.equal(target.candidate,false);assert.match(target.note,/自分で/);
});
test('selected target is carried with exact source revision in URL and upload',()=>{
  const target={run:'run-1',result:'result-2',key:'surface:wall-3'};
  const url=new URL(targetPhotoUrl('parent',target.run,target.result,target.key),'http://localhost');assert.equal(url.searchParams.get('target'),target.key);assert.equal(url.searchParams.get('result'),target.result);
  const body=revisionBody({scene:{id:'parent',name:'room'},quality:{options:{up_axis:'image-top',max_frames:60,furniture_frame_budget:24,texture_quality:'high',refine_cameras:false}},photos:[new File(['photo'],'p.jpg')],captureTarget:target});
  assert.deepEqual(JSON.parse(body.get('capture_target')),target);assert.equal(body.get('parent_scene_id'),'parent');
});
