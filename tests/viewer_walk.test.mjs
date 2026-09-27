import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import * as THREE from '../web/vendor/three.module.js';

function setup() {
  const window=new EventTarget(), document=new EventTarget(), canvas=new EventTarget();
  document.querySelectorAll=()=>[];
  canvas.style={}; canvas.setPointerCapture=()=>{};
  const source=readFileSync(new URL('../web/walk-controls.js', import.meta.url),'utf8')
    .replace("import * as THREE from 'three';", '').replace('export class WalkControls', 'class WalkControls');
  const context=vm.createContext({THREE, window, document, AbortController});
  const Controller=vm.runInContext(source+'\nWalkControls',context);
  const camera=new THREE.PerspectiveCamera();
  camera.up.set(0,0,1); camera.lookAt(0,1,0);
  const controls={target:new THREE.Vector3(),enabled:true};
  const walk=new Controller(camera,canvas,()=>controls,()=>walk.stop());
  walk.start(10);
  return {walk,camera,controls,window};
}

test('walking maintains height in a rotated VGGT coordinate system',()=>{
  const {walk,camera}=setup();
  walk.keys.add('KeyW'); walk.update(0); walk.update(50);
  assert.ok(camera.position.y>0);
  assert.ok(Math.abs(camera.position.z)<1e-10);
});
test('diagonal movement is normalized and stalled frames cannot teleport',()=>{
  const {walk,camera}=setup();
  walk.keys.add('KeyW'); walk.keys.add('KeyD'); walk.update(0); walk.update(60000);
  assert.ok(Math.abs(camera.position.length()-.04)<1e-10);
});
test('losing focus stops movement and ending restores orbit controls',()=>{
  const {walk,camera,controls,window}=setup();
  walk.keys.add('KeyW'); walk.update(0);
  window.dispatchEvent(new Event('blur')); walk.update(50); walk.update(100);
  assert.equal(camera.position.length(),0);
  walk.stop(); assert.equal(controls.enabled,true); assert.equal(walk.active,false);
});

test('movement constraint is applied to the actual camera, including vertical keys',()=>{
  const {walk,camera}=setup();
  walk.constrainMove=(p,d)=>({x:p.x+d.x,y:Math.min(.1,p.y+d.y),z:1.5});
  walk.keys.add('KeyW');walk.keys.add('KeyE');
  for(let frame=0;frame<120;frame++)walk.update(frame*16);
  assert.equal(camera.position.y,.1);assert.equal(camera.position.z,1.5);
});
