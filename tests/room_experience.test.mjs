import test from 'node:test';
import assert from 'node:assert/strict';
import {generationPhase} from '../web/generation-progress.mjs';
import {findRoomRoute} from '../web/room-route.mjs';
import {RoomNavigation} from '../web/room-navigation.js';
const room={footprint:[[0,0],[6,0],[6,6],[0,6]],floor:0,ceiling:2.5,radius:.18,eye_height:1.5};
const box={center:[3,3,.5],scale:[1,2,1],rotation:[[1,0,0],[0,1,0],[0,0,1]],bottom:0,top:1};
test('capture and room jobs map actual milestones without reaching 100 early',()=>{
  assert.equal(generationPhase({state:'running',stage:'capture'}).index,0);
  const percentages=[];for(let step=1;step<=6;step++){const p=generationPhase({state:'running',message:`${step}/6 処理中`});assert.ok(p.percent<100);percentages.push(p.percent);}
  assert.deepEqual(percentages,[40,52,64,76,86,94]);assert.equal(generationPhase({state:'complete'}).percent,100);
  assert.equal(generationPhase({state:'failed',message:'6/6 failed'}).failed,true);
  assert.equal(generationPhase({state:'interrupted'}).failed,true);
});
test('guide bends around furniture and every segment remains walkable',()=>{
  const nav=new RoomNavigation({...room,colliders:[box]});const points=findRoomRoute(nav,{x:1,y:3},{x:5,y:3});
  assert.ok(points&&points.length>2);
  for(let i=1;i<points.length;i++)for(let j=0;j<=100;j++){const a=points[i-1],b=points[i];assert.ok(nav.valid(a.x+(b.x-a.x)*j/100,a.y+(b.y-a.y)*j/100));}
  assert.ok(Math.hypot(points.at(-1).x-5,points.at(-1).y-3)<.3);
});
test('blocked and outside targets cannot produce an invented straight route',()=>{
  const nav=new RoomNavigation({...room,colliders:[{...box,scale:[1,6,1]}]});
  assert.equal(findRoomRoute(nav,{x:1,y:3},{x:5,y:3}),null);
  assert.equal(findRoomRoute(nav,{x:1,y:3},{x:100,y:100}),null);
  assert.equal(findRoomRoute(nav,{x:-1,y:3},{x:2,y:3}),null);
});
test('concave floor route stays within the observed footprint',()=>{
  const nav=new RoomNavigation({...room,footprint:[[0,0],[5,0],[5,1.5],[1.5,1.5],[1.5,5],[0,5]]});
  const points=findRoomRoute(nav,{x:4,y:.7},{x:.7,y:4});assert.ok(points?.length>=3);
  for(let i=1;i<points.length;i++)for(let j=0;j<=100;j++){const a=points[i-1],b=points[i];assert.ok(nav.valid(a.x+(b.x-a.x)*j/100,a.y+(b.y-a.y)*j/100));}
});
