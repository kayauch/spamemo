import test from 'node:test';
import assert from 'node:assert/strict';
import {photoAdvice,revisionBody} from '../web/photo-revision.mjs';
const options={room_type:'storage',up_axis:'image-right',ceiling_height_m:2.7,reference_length_m:5,max_frames:60,furniture_frame_budget:24,texture_quality:'high',refine_cameras:true,furniture_overrides:[{id:'old-id',hidden:true}]};
const scene={id:'parent-room',name:'元の部屋',reconstruction_model:'VGGT-1B'};
const photo=new File(['fixture'],'wall.jpg',{type:'image/jpeg'});
test('photo revision always carries parent, original calibration, and only new photos',()=>{
  const body=revisionBody({scene,quality:{options},photos:[photo]});
  assert.equal(body.get('parent_scene_id'),scene.id);assert.equal(body.get('name'),scene.name);
  assert.equal(body.getAll('photos').length,1);assert.equal(body.get('photos').name,'wall.jpg');assert.equal(body.has('videos'),false);
  const result=JSON.parse(body.get('options'));
  for(const key of ['up_axis','ceiling_height_m','reference_length_m','max_frames','texture_quality','refine_cameras'])assert.equal(result[key],options[key]);
  assert.equal(Object.hasOwn(result,'furniture_overrides'),false,'old object IDs must not affect re-estimated furniture');
});
test('cannot create an unrelated room or submit empty/incompatible data',()=>{
  assert.throws(()=>revisionBody({scene:{},quality:{options},photos:[photo]}),/追加先/);
  assert.throws(()=>revisionBody({scene,quality:{options},photos:[]}),/選んで/);
  assert.throws(()=>revisionBody({scene:{...scene,reconstruction_model:'calibrated-rgbd'},quality:{options},photos:[photo]}),/深度/);
  assert.throws(()=>revisionBody({scene,quality:{},photos:[photo]}),/向き/);
});
test('geometry mismatches and hidden furniture are not reported as missing photos',()=>{
  const quality={objects:[{id:1,label:'shelf'},{id:2,label:'chair',hidden:true}],recommendations:[
    {kind:'floor',code:'projection_depth_mismatch'},
    {kind:'wall',code:'furniture_mask_occlusion'},
    {kind:'ceiling',code:'missing_photo_coverage'},
    {kind:'furniture',object_id:1,code:'storage_side_depth_unresolved'},
    {kind:'furniture',object_id:2,code:'full_object_photo'}]};
  const advice=photoAdvice(quality);assert.equal(advice.needsRepair,true);assert.equal(advice.tips.length,2);
  assert.match(advice.tips[0],/天井/);assert.match(advice.tips[1],/棚/);assert.doesNotMatch(advice.tips.join(),/椅子/);
});
