import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../web/vendor/three.module.js';
import {createStorageHighlight} from '../web/storage-highlight.mjs';

test('preview highlights transformed furniture without changing shared materials or geometry',()=>{
 const scene=new THREE.Scene(),parent=new THREE.Group();parent.position.set(4,2,1);parent.rotation.z=Math.PI/2;scene.add(parent);
 const material=new THREE.MeshBasicMaterial({transparent:true,opacity:.14,depthWrite:false});
 const geometry=new THREE.BoxGeometry(1,2,3),mesh=new THREE.Mesh(geometry,material);parent.add(mesh);
 let sourceDisposed=false;geometry.addEventListener('dispose',()=>sourceDisposed=true);
 const highlighter=createStorageHighlight(scene),bounds=highlighter.show([parent,mesh]);
 const overlays=scene.getObjectByName('storage-selection');assert.equal(overlays.children.length,2);
 assert.deepEqual(bounds.getCenter(new THREE.Vector3()).toArray(),[4,2,1]);
 assert.notEqual(overlays.children[0].geometry,geometry);assert.equal(overlays.children[0].material.depthTest,false);
 assert.equal(mesh.material,material);assert.equal(material.opacity,.14);assert.equal(material.depthWrite,false);
 highlighter.clear();assert.equal(overlays.children.length,0);assert.equal(sourceDisposed,false);
 assert.equal(parent.children[0],mesh);highlighter.dispose();assert.equal(scene.getObjectByName('storage-selection'),undefined);
});

test('manual storage has a bounded rotated preview and changing selection removes old marks',()=>{
 const scene=new THREE.Scene(),highlighter=createStorageHighlight(scene);
 const bounds=highlighter.show([],{room_position:[1,2,0],width:80,depth:30,height:180,angle:90});
 const size=bounds.getSize(new THREE.Vector3());assert.ok(Math.abs(size.x-.3)<1e-6);assert.ok(Math.abs(size.y-.8)<1e-6);assert.ok(Math.abs(size.z-1.8)<1e-6);
 assert.equal(scene.getObjectByName('storage-selection').children.length,2);
 assert.ok(highlighter.show([],null).isEmpty());assert.equal(scene.getObjectByName('storage-selection').children.length,0);
 assert.ok(highlighter.show([],{room_position:[0,0,0],width:NaN,depth:30,height:180}).isEmpty());highlighter.dispose();
});
