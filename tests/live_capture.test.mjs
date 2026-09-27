import test from 'node:test';
import assert from 'node:assert/strict';
import {LiveCaptureSession,guidedRevisionBody} from '../web/live-capture-session.mjs';
const targets=[{key:'surface:wall-1',title:'壁 1'},{key:'surface:floor',title:'床'},{key:'object:7',title:'棚'}];
const file=name=>new File(['photo'],name,{type:'image/jpeg'});
test('successful capture advances exactly once and ends without wrapping',()=>{
  const s=new LiveCaptureSession(targets);assert.equal(s.current,targets[0]);
  assert.throws(()=>s.commit(new File([],'empty.jpg')));assert.equal(s.index,0);
  for(let i=0;i<3;i++){s.commit(file(i+'.jpg'));assert.equal(s.index,i+1);}
  assert.equal(s.current,null);assert.equal(s.photos.length,3);assert.throws(()=>s.commit(file('extra.jpg')));
});
test('skip creates no photo; retake preserves old photo until successful replacement',()=>{
  const s=new LiveCaptureSession(targets),first=file('old.jpg');s.commit(first);s.skip();s.commit(file('shelf.jpg'));
  assert.deepEqual(s.photos.map(p=>p.target.key),[targets[0].key,targets[2].key]);
  s.retry(targets[0].key);assert.throws(()=>s.commit(null));assert.equal(s.photos[0].file,first);
  s.commit(file('new.jpg'));assert.equal(s.current,null);assert.equal(s.photos.length,2);assert.equal(s.photos[0].file.name,'new.jpg');
  s.resume();assert.equal(s.current.key,targets[1].key);
  s.commit(file('floor.jpg'));assert.equal(s.current,null);assert.equal(s.photos[2].file.name,'shelf.jpg');
});
test('batch upload keeps each photo aligned with its exact target after retakes and skips',()=>{
  const s=new LiveCaptureSession(targets);s.skip();s.commit(file('floor.jpg'));s.commit(file('shelf.jpg'));
  const body=guidedRevisionBody({scene:{id:'parent',name:'room'},quality:{options:{up_axis:'image-top',max_frames:60,furniture_frame_budget:24,texture_quality:'high'}},session:s,run:'run',result:'result'});
  assert.deepEqual(body.getAll('photos').map(f=>f.name),['floor.jpg','shelf.jpg']);
  assert.deepEqual(JSON.parse(body.get('capture_targets')).map(t=>t.key),['surface:floor','object:7']);
  assert.equal(body.has('capture_target'),false);assert.equal(body.get('parent_scene_id'),'parent');
});
