import test from 'node:test';
import assert from 'node:assert/strict';
import {repairCapturePlan,photoOrientation,MAX_REPAIR_PHOTOS} from '../web/repair-capture-plan.mjs';
import {captureTargets} from '../web/capture-targets.mjs';
import {LiveCaptureSession,guidedRevisionBody} from '../web/live-capture-session.mjs';
import {validateCaptureFiles} from '../web/capture-utils.mjs';
const file=name=>new File(['photo'],name,{type:'image/jpeg'});

test('repair plan covers every unknown structure and furniture with distinct overlapping shots',()=>{
  const room={surfaces:[{id:'floor',kind:'floor',origin:[0,0,0],u:[4,0,0],v:[0,4,0]},{id:'ceiling',kind:'ceiling'},{id:'wall-1',kind:'wall'},{id:'a',object_id:7,object_label:'cabinet'}],furniture:[{id:7,label:'cabinet'},{id:8,label:'chair',hidden:true}]};
  const plan=repairCapturePlan(captureTargets(room));
  assert.equal(plan.length,15);assert.equal(new Set(plan.map(t=>t.shotId)).size,15);
  assert.deepEqual([...new Set(plan.map(t=>t.key))],['surface:floor','surface:ceiling','surface:wall-1','object:7']);
  assert.ok(plan[0].region.u1>plan[1].region.u0);
  assert.ok(plan[0].region.v1>plan[2].region.v0);
  assert.ok(plan.every(t=>t.instruction.includes('横向き')));
  assert.ok(plan.filter(t=>t.kind==='object').every(t=>t.instruction.includes('家具')));
});
test('known covered surfaces are skipped unless explicitly selected; mixed geometry warnings are retained',()=>{
  const room={surfaces:[{id:'wall-1',kind:'wall',supported_fraction:.98},{id:'wall-2',kind:'wall',supported_fraction:.2}]};
  const quality={recommendations:[{surface_id:'wall-2',code:'projection_depth_mismatch'}]};
  const targets=captureTargets(room,quality),plan=repairCapturePlan(targets);
  assert.ok(plan.every(t=>t.key==='surface:wall-2'&&t.repair));
  const selected=repairCapturePlan(targets,{selectedKey:'surface:wall-1'});
  assert.equal(selected[0].key,'surface:wall-1');assert.equal(selected.length,8);
});
test('long wall gets more coverage cells without changing canonical upload target',()=>{
  const plan=repairCapturePlan([{key:'surface:wall-1',kind:'wall',title:'壁 1',shape:{u:[12,0,0],v:[0,0,2.5]}}]);
  assert.equal(plan.length,10);assert.ok(plan.every(t=>t.key==='surface:wall-1'));
  assert.equal(plan.at(-1).region.u1,1);assert.equal(plan.at(-1).region.v1,1);
});
test('multiple angles of the same wall never overwrite each other; retry replaces only one shot',()=>{
  const targets=repairCapturePlan([{key:'surface:wall-1',kind:'wall',title:'壁 1'}]);
  const session=new LiveCaptureSession(targets);session.commit(file('a.jpg'));session.commit(file('b.jpg'));session.skip();session.commit(file('d.jpg'));
  assert.equal(session.photos.length,3);assert.equal(session.pending.length,1);
  session.retry(targets[1].shotId);session.commit(file('retry.jpg'));
  assert.deepEqual(session.photos.map(p=>p.file.name),['a.jpg','retry.jpg','d.jpg']);
  session.resume();assert.equal(session.current.shotId,targets[2].shotId);session.commit(file('c.jpg'));
  const body=guidedRevisionBody({scene:{id:'parent',name:'room'},quality:{options:{up_axis:'image-top',max_frames:60,furniture_frame_budget:24,texture_quality:'high'}},session,run:'run',result:'result'});
  assert.equal(body.getAll('photos').length,4);assert.ok(JSON.parse(body.get('capture_targets')).every(t=>t.key==='surface:wall-1'));
});
test('guide keeps more than the upload cap visible instead of silently truncating targets',()=>{
  const targets=Array.from({length:70},(_,i)=>({key:'surface:wall-1',shotId:'shot-'+i}));
  const session=new LiveCaptureSession(targets);assert.equal(session.targets.length,70);
  session.addTargets([{key:'object:8',shotId:'new'},{key:'surface:wall-1',shotId:'shot-1'}]);assert.equal(session.targets.length,71);
  assert.equal(MAX_REPAIR_PHOTOS,64);
  validateCaptureFiles({photos:Array.from({length:64},(_,i)=>file(i+'.jpg'))});
  assert.throws(()=>validateCaptureFiles({photos:Array.from({length:65},(_,i)=>file(i+'.jpg'))}),/64/);
});
test('a room with many objects gets every object overview before secondary angles',()=>{
  const targets=[...Array.from({length:6},(_,i)=>({key:'surface:'+i,kind:'wall',title:'壁'})),...Array.from({length:19},(_,i)=>({key:'object:'+i,kind:'object',title:'家具'}))];
  const plan=repairCapturePlan(targets);
  assert.equal(plan.length,81);assert.equal(new Set(plan.slice(0,64).map(t=>t.key)).size,25);
  const firstSide=plan.findIndex(t=>t.kind==='object'&&t.shotIndex>0);
  assert.ok(plan.slice(0,firstSide).filter(t=>t.kind==='object').every(t=>t.shotIndex===0));
  assert.equal(plan.slice(0,firstSide).filter(t=>t.kind==='object').length,19);
});
test('portrait viewport, portrait image and unloaded stream cannot pass landscape guard',()=>{
  assert.equal(photoOrientation({landscape:false,width:1920,height:1080}).allowed,false);
  assert.equal(photoOrientation({landscape:true,width:1080,height:1920}).allowed,false);
  assert.equal(photoOrientation({landscape:true,width:0,height:0}).allowed,false);
  assert.equal(photoOrientation({landscape:true,width:1920,height:1080}).allowed,true);
});
