import test from 'node:test';
import assert from 'node:assert/strict';
import {resolveCaptureEntry,isMissingCaptureParent} from '../web/capture-entry.mjs';

test('deleted room links become new captures; existing parents are retained',()=>{
  assert.deepEqual(resolveCaptureEntry([], 'deleted'),{sceneId:null,missing:true});
  assert.deepEqual(resolveCaptureEntry([{id:'kept'}], 'kept'),{sceneId:'kept',missing:false});
  assert.deepEqual(resolveCaptureEntry([], null),{sceneId:null,missing:false});
});
test('unavailable scene lists cannot be treated as deleted rooms',()=>{
  assert.throws(()=>resolveCaptureEntry(null,'kept'));
  assert.throws(()=>resolveCaptureEntry({error:'unavailable'},'kept'));
});
test('only the explicit missing-parent response allows parent recovery',()=>{
  assert.equal(isMissingCaptureParent({status:422,message:'追加先の部屋が見つかりません'}),true);
  assert.equal(isMissingCaptureParent({status:422,message:'空のファイルは使えません'}),false);
  assert.equal(isMissingCaptureParent({status:500,message:'追加先の部屋が見つかりません'}),false);
});
