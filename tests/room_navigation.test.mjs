import test from 'node:test';
import assert from 'node:assert/strict';
import {RoomNavigation} from '../web/room-navigation.js';
const room={footprint:[[0,0],[4,0],[4,4],[0,4]],floor:0,ceiling:2.5,radius:.18,eye_height:1.5};
const box={center:[2,2,.5],scale:[1,1,1],rotation:[[1,0,0],[0,1,0],[0,0,1]],bottom:0,top:1};
test('large movement cannot tunnel through walls and keeps floor height',()=>{
  const nav=new RoomNavigation(room);
  const p=nav.move({x:1,y:1,z:1.5},{x:100,y:0,z:50});
  assert.ok(p.x<=3.82&&p.x>3.6);assert.equal(p.z,1.5);
});
test('blocks furniture and slides along it instead of entering it',()=>{
  const nav=new RoomNavigation({...room,colliders:[box]});
  const p=nav.move({x:1,y:2,z:1.5},{x:2,y:0,z:0});
  assert.ok(p.x<1.33);assert.ok(nav.valid(p.x,p.y));
  const q=nav.move(p,{x:.2,y:.2,z:0});assert.ok(q.y>p.y);
});
test('concave recess is not walkable; spawn avoids furniture',()=>{
  const nav=new RoomNavigation({...room,footprint:[[0,0],[4,0],[4,1],[1,1],[1,4],[0,4]]});
  assert.equal(nav.valid(2,2),false);
  const p=nav.findSpawn([2,2]);assert.ok(nav.valid(p.x,p.y));
  const furniture=new RoomNavigation({...room,colliders:[box]});
  const q=furniture.findSpawn([2,2]);assert.ok(furniture.valid(q.x,q.y));
});
test('rotated furniture uses its actual footprint',()=>{
  const t=Math.PI/4,c=Math.cos(t),s=Math.sin(t);
  const nav=new RoomNavigation({...room,colliders:[{...box,scale:[2,.2,1],rotation:[[c,-s,0],[s,c,0],[0,0,1]]}]});
  assert.equal(nav.valid(2.6,2.6),false);assert.equal(nav.valid(2.6,1.4),true);
});
test('reports no spawn when room is completely occupied',()=>{
  const nav=new RoomNavigation({...room,colliders:[{...box,scale:[10,10,1]}]});
  assert.equal(nav.findSpawn(),null);
});
