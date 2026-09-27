import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const code=readFileSync(new URL('../web/splat-layer.js',import.meta.url),'utf8');
const {SplatLayer}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
function fixture() {
  const pending=[], viewers=[], objects=new Set();
  const GS={SplatRenderMode:{TwoD:2}, SceneRevealMode:{Instant:0}, SceneFormat:{Ply:1},
    DropInViewer:class {
      constructor(options){this.options=options;viewers.push(this);this.disposed=false;}
      addSplatScene(url,options){this.url=url;this.sceneOptions=options;return new Promise((resolve,reject)=>pending.push({resolve,reject}));}
      async dispose(){this.disposed=true;}
    }};
  return {pending,viewers,objects,layer:new SplatLayer({add:v=>objects.add(v),remove:v=>objects.delete(v)},async()=>GS)};
}
test('2D mode uses identity transform and releases old model',async()=>{
  const f=fixture();const loading=f.layer.load('/model.ply');await Promise.resolve();
  f.pending[0].resolve();assert.equal(await loading,true);
  assert.equal(f.viewers[0].options.splatRenderMode,2);
  assert.deepEqual(f.viewers[0].sceneOptions.rotation,[0,0,0,1]);
  assert.deepEqual(f.viewers[0].sceneOptions.scale,[1,1,1]);
  f.layer.setVisible(false);assert.equal(f.viewers[0].visible,false);
  f.layer.clear();assert.equal(f.objects.size,0);assert.equal(f.viewers[0].disposed,true);
});
test('switching away during download cannot resurrect a splat model',async()=>{
  const f=fixture();const loading=f.layer.load('/slow.ply');await Promise.resolve();
  f.layer.clear();f.pending[0].resolve();assert.equal(await loading,false);
  assert.equal(f.objects.size,0);assert.equal(f.viewers[0].disposed,true);
});
test('download error releases worker and GPU resources',async()=>{
  const f=fixture();const loading=f.layer.load('/missing.ply');await Promise.resolve();
  f.pending[0].reject(new Error('404'));await assert.rejects(loading,/404/);
  assert.equal(f.objects.size,0);assert.equal(f.viewers[0].disposed,true);
});
