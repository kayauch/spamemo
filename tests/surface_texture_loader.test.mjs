import test from 'node:test';
import assert from 'node:assert/strict';
import {loadSurfaceTextures} from '../web/surface-texture-loader.mjs';

const surfaces=Array.from({length:6},(_,i)=>({texture:`photo-${i}`,support:`mask-${i}`}));
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};}
const tick=()=>new Promise(resolve=>setImmediate(resolve));

test('request concurrency is bounded and out-of-order completion preserves photo/support pairs',async()=>{
  let active=0,peak=0;const requests=[],progress=[];
  const pending=loadSurfaceTextures(surfaces,path=>{
    const request=deferred();requests.push({path,...request});active++;peak=Math.max(peak,active);
    return request.promise.finally(()=>active--);
  },{concurrency:4,onProgress:(done,total)=>progress.push([done,total])});
  assert.equal(requests.length,4);assert.equal(active,4);
  // Resolve in reverse order within each batch, including the mask before photo.
  for(let start=0;start<12;start+=4){
    for(const request of requests.slice(start,start+4).reverse())request.resolve({path:request.path});
    await tick();
    assert.ok(active<=4);
  }
  const pairs=await pending;
  assert.equal(peak,4);assert.equal(requests.length,12);assert.equal(active,0);
  assert.deepEqual(pairs,surfaces.map(surface=>({texture:{path:surface.texture},mask:{path:surface.support}})));
  assert.deepEqual(progress,Array.from({length:12},(_,i)=>[i+1,12]));
});

test('failure stops queued work and disposes successful assets including late arrivals',async()=>{
  const requests=[],disposed=[];
  const pending=loadSurfaceTextures(surfaces,path=>{
    const request=deferred();requests.push({path,...request});return request.promise;
  },{concurrency:4});
  const rejected=assert.rejects(pending,/photo unavailable/);
  requests[1].reject(new Error('photo unavailable'));
  await tick();
  assert.equal(requests.length,4);
  requests[0].resolve({dispose:()=>disposed.push(0)});
  requests[2].reject(new Error('secondary error'));
  requests[3].resolve({dispose:()=>disposed.push(3)});
  await rejected;
  assert.equal(requests.length,4);assert.deepEqual(disposed.sort(),[0,3]);
});

test('successful assets are retained; empty input and invalid bounds are handled',async()=>{
  let disposed=0;
  const pairs=await loadSurfaceTextures(surfaces.slice(0,1),async path=>({path,dispose:()=>disposed++}));
  assert.equal(pairs.length,1);assert.equal(disposed,0);
  assert.deepEqual(await loadSurfaceTextures([],()=>{throw Error('unexpected');}),[]);
  await assert.rejects(loadSurfaceTextures(surfaces,()=>{}, {concurrency:0}),RangeError);
  await assert.rejects(loadSurfaceTextures(surfaces,()=>{}, {concurrency:1.5}),RangeError);
});
