import test from 'node:test';
import assert from 'node:assert/strict';
import {capturePlan,guidedInstruction,captureReceipt} from '../web/capture-guide.mjs';
test('geometry mismatch must not become a request for more photos',()=>{
 const plan=capturePlan({recommendations:[{kind:'wall',surface_id:'w1',code:'missing_photo_coverage'},{kind:'wall',surface_id:'w1',code:'projection_depth_mismatch',message:'check geometry'}]});
 assert.equal(plan.tasks.length,0);assert.deepEqual(plan.repairs,['check geometry']);
});
test('furniture guide uses actual source references and retains uncertain candidates',()=>{
 const plan=capturePlan({objects:[{id:6022,label:'wardrobe',review_status:'needs_confirmation',source_frames:[49]}]});
 assert.equal(plan.tasks.length,1);assert.deepEqual(plan.tasks[0].frames,[49]);assert.match(plan.tasks[0].instruction,/斜め/);
});
test('structure instructions do not invent a matching source photo',()=>{
 const plan=capturePlan({recommendations:[{kind:'floor',surface_id:'floor',code:'missing_photo_coverage'},{kind:'floor',surface_id:'floor',code:'missing_photo_coverage'}],frames:[{index:1,url:'/unrelated'}]});
 assert.equal(plan.tasks.length,1);assert.deepEqual(plan.tasks[0].frames,[]);
});

test('camera guide gives one instruction per photo and never asks to move furniture',()=>{
 const task={id:'object-7'};
 assert.match(guidedInstruction(task,0),/家具全体/);
 assert.match(guidedInstruction(task,1),/家具は動かさず/);
 assert.equal(guidedInstruction({id:'floor',instruction:'floor prompt'},0),'floor prompt');
});

test('receipt distinguishes captured photos from upload completion',()=>{
 assert.match(captureReceipt(0),/まだ撮影していません/);
 assert.match(captureReceipt(2),/2枚・未送信/);
});
