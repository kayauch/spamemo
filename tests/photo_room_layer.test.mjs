import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import * as THREE from '../web/vendor/three.module.js';
import {RoomNavigation} from '../web/room-navigation.js';

const source=readFileSync(new URL('../web/photo-room-layer.js',import.meta.url),'utf8')
  .replace(/^import .*;$/gm,'').replaceAll('export class ','class ').replaceAll('export function ','function ');
function classes(overrides={}){
  return vm.runInNewContext(source+'\n({PhotoRoomFrame,PhotoRoomLayer,surfaceGeometry})',{THREE:{...THREE,...overrides},RoomNavigation,URL});
}
const {PhotoRoomFrame,surfaceGeometry}=classes();
const surface={id:'wall',kind:'wall',origin:[0,0,0],u:[2,0,0],v:[0,0,2],texture:'wall.jpg'};
function data(){
  // Room Z-up becomes VGGT X-up, with translation and a non-unit scale.
  return {schema_version:2,transform:{scene_id:'test-scene',
    generation_world_to_vg_world:[[0,0,2,3],[2,0,0,-4],[0,2,0,8],[0,0,0,1]]},
    navigation:{floor:0,ceiling:3,footprint:[[0,0],[4,0],[4,4],[0,4]],radius:.18,eye_height:1.5},
    surfaces:[surface]};
}
function near(a,b){assert.ok(a.distanceTo(b)<1e-6,JSON.stringify([a,b]));}

test('same room point round-trips to original API coordinates under rotation/scale/translation',()=>{
  const frame=new PhotoRoomFrame(data(),'test-scene');
  const room=new THREE.Vector3(1,2,3),world=frame.world(room);
  near(world,new THREE.Vector3(9,-2,12));near(frame.room(world),room);
  near(frame.up,new THREE.Vector3(1,0,0));assert.equal(frame.scale,2);
});

test('rejects another scene and invalid/nonuniform transforms',()=>{
  assert.throws(()=>new PhotoRoomFrame(data(),'another-scene'));
  const singular=data();singular.transform.generation_world_to_vg_world[0]=[0,0,0,0];
  assert.throws(()=>new PhotoRoomFrame(singular,'test-scene'));
  const unequal=data();unequal.transform.generation_world_to_vg_world[0][2]=3;
  assert.throws(()=>new PhotoRoomFrame(unequal,'test-scene'));
});

test('raycast on a photo wall returns the exact position to save in the original coordinate frame',()=>{
  const frame=new PhotoRoomFrame(data(),'test-scene'),group=new THREE.Group();
  group.matrixAutoUpdate=false;group.matrix.copy(frame.toWorld);
  const mesh=new THREE.Mesh(surfaceGeometry(surface),new THREE.MeshBasicMaterial({side:THREE.DoubleSide}));
  group.add(mesh);group.updateMatrixWorld(true);
  const origin=frame.world(new THREE.Vector3(1,-2,1));
  const direction=new THREE.Vector3(0,1,0).transformDirection(frame.toWorld);
  const hit=new THREE.Raycaster(origin,direction).intersectObject(group,true)[0];
  assert.ok(hit);near(hit.point,new THREE.Vector3(5,-2,8));
  near(frame.room(hit.point),new THREE.Vector3(1,0,1));
  mesh.geometry.dispose();mesh.material.dispose();
});

test('walking collision and eye height still hold in the original rotated coordinate system',()=>{
  const frame=new PhotoRoomFrame(data(),'test-scene'),nav=new RoomNavigation(data().navigation);
  const start=frame.world(new THREE.Vector3(2,2,1.5));
  const next=frame.constrain(nav,start,new THREE.Vector3(12,40,0));
  const room=frame.room(next);
  assert.equal(room.z,1.5);assert.ok(room.x<4&&room.x>=3.7);assert.equal(room.y,2);
});

test('cancelled photo loading disposes the texture and never attaches a stale model',async()=>{
  let resolve;const texture=new THREE.Texture();let disposed=0;texture.addEventListener('dispose',()=>disposed++);
  const {PhotoRoomLayer}=classes({TextureLoader:class{loadAsync(){return new Promise(r=>resolve=r);}}});
  const scene=new THREE.Scene(),layer=new PhotoRoomLayer(scene,'test-scene',{capabilities:{getMaxAnisotropy:()=>8}});
  layer.configure(data(),'http://localhost/assets/');
  const pending=layer.load();layer.clear();resolve(texture);
  assert.equal(await pending,false);assert.equal(scene.children.length,0);assert.equal(disposed,1);
});

test('a missing photo fails atomically and releases all partial resources',async()=>{
  const first=new THREE.Texture();let disposed=0,n=0;first.addEventListener('dispose',()=>disposed++);
  const {PhotoRoomLayer}=classes({TextureLoader:class{async loadAsync(){if(n++===0)return first;throw Error('missing');}}});
  const scene=new THREE.Scene(),layer=new PhotoRoomLayer(scene,'test-scene',{capabilities:{getMaxAnisotropy:()=>8}});
  const model=data();model.surfaces.push({...surface,id:'second'});
  layer.configure(model,'http://localhost/assets/');
  await assert.rejects(layer.load(),/missing/);
  assert.equal(scene.children.length,0);assert.equal(disposed,1);assert.equal(layer.pending,null);
});

test('registration posts the raycast world position without applying a second transform',async()=>{
  const app=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
  const begin=app.indexOf('async function save3dSelection()');
  const end=app.indexOf('const geometryVariantSelect',begin);
  const controls={selectionName:{value:'Test item'},save3dSelectionBtn:{},selectionStatus:{},answer:{}};
  let payload;
  const context=vm.createContext({
    sceneId:'test-scene',selected3dPosition:new THREE.Vector3(5,-2,8),
    selected3dSource:{type:'photo-room',surfaceId:'wall'},
    document:{getElementById:id=>controls[id]},fetch:async(url,options)=>{payload=JSON.parse(options.body);return {};},
    safeJson:async()=>({id:'new-id',name:'Test item',position:{x:5,y:-2,z:8}}),
    refresh:async()=>{},showSelectionMarker:()=>{},vec:p=>new THREE.Vector3(p.x,p.y,p.z)
  });
  await vm.runInContext(app.slice(begin,end)+'\nsave3dSelection()',context);
  assert.deepEqual(payload.position,{x:5,y:-2,z:8});
  assert.equal(payload.metadata.coordinate_frame,'vggt_world');
  assert.equal(payload.metadata.photo_room_surface,'wall');
});
