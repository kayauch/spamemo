import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {PLYLoader} from './vendor/PLYLoader.js';

const $=id=>document.getElementById(id);
const run=new URLSearchParams(location.search).get('run');
const prediction=new URLSearchParams(location.search).get('prediction');
const colors={wall:0x67b5ff,door:0xffb961,window:0x64ead2,wardrobe:0xd3a5ff,cupboard:0xd3a5ff,wall_cabinet:0xd3a5ff,chair:0xffdf6b,dining_chair:0xffdf6b};
const labels={locker:'ロッカー候補',shelf:'棚候補',box:'収納箱候補',cabinet:'収納候補',bookshelf:'本棚候補',wall:'壁の区間',door:'扉候補',window:'窓候補',wardrobe:'収納候補',cupboard:'収納候補',wall_cabinet:'収納候補',chair:'椅子候補',dining_chair:'椅子候補',curtain:'カーテン候補',plants:'植物候補',illumination:'照明候補',chandelier:'照明候補'};
const colorFor=name=>colors[name]??0xb6c7f1;
try{
  if(!run)throw new Error('確認する処理が指定されていません。作成した部屋の比較結果から開いてください。');
  if(!/^[a-zA-Z0-9_-]+$/.test(run))throw new Error('実験名が不正です');
  if(prediction&&!/^prediction-[a-zA-Z0-9_-]+$/.test(prediction))throw new Error('比較対象が不正です');
  const base=`/static/generation-reviews/${run}/${prediction?prediction+'/':''}`;
  const response=await fetch(`${base}review.json`);
  if(!response.ok)throw new Error(`結果を読み込めません (${response.status})`);
  const data=await response.json();
  const indexResponse=await fetch(`/static/generation-reviews/${run}/index.json`);
  if(indexResponse.ok){for(const entry of await indexResponse.json()){const option=document.createElement('option');option.value=entry.id;option.textContent=entry.label;option.selected=entry.id===data.prediction_id;$('predictions').append(option);}}
  $('predictions').onchange=event=>{const url=new URL(location.href);url.searchParams.set('prediction',event.target.value);location.href=url.href;};
  $('json').href=`${base}review.json`;
  const solidResponse=await fetch(`/static/generation-reviews/${run}/solid-photo-001/room.json`);
  if(solidResponse.ok){
    const solid=await solidResponse.json();
    if(solid.source_prediction===data.prediction_id){
      const link=document.createElement('a');link.href=`/static/solid-room.html?run=${run}&result=solid-photo-001`;
      link.textContent='点群なし・写真を貼った部屋を見る';link.style.display='block';link.style.margin='12px 0';$('json').before(link);
    }
  }
  $('status').textContent=data.prediction?'実験結果・未採用 — 実写との照合が必要':'入力の準備完了 — AI推定待ち';
  if(data.prediction?.photo_edge_corrections){
    const removed=data.prediction.rejected_proposals?.length??0;
    const corrected=data.prediction.photo_edge_corrections.filter(c=>c.accepted).length;
    $('status').textContent=`実写照合による修正案：扉候補${removed}件を除外、収納${corrected}列の上端を補正（未確定）`;
    const note=document.createElement('p');note.className='small';note.textContent='上端は実写で手動指定し、奥行きで高さを算出。別画像で一致しない補正は保留しています。幅・厚さ・区画数は未確認です。';$('counts').after(note);
  }
  $('scale').textContent=`寸法は天井高 ${data.transform.assumed_height_m}m を仮定した換算です。実測値ではありません。`;
  $('orientation').textContent=data.transform.up_sign_flipped_from_existing_frame?'実験用の上下方向を補正しています。既存モデルの床・天井ラベルは要確認です。':'';
  const displayCounts=new Map();
  for(const [category,count] of Object.entries(data.category_counts)){const name=labels[category]??category;const entry=displayCounts.get(name)??{category,count:0};entry.count+=count;displayCounts.set(name,entry);}
  $('counts').textContent=data.prediction?[...displayCounts].map(([name,entry])=>`${name}: ${entry.count}`).join(' ／ '):'推定結果はまだありません';
  for(const [name,entry] of displayCounts){
    const item=document.createElement('span');item.textContent=`${name} ${entry.count}`;item.style.borderColor=`#${colorFor(entry.category).toString(16)}`;$('legend').append(item);
  }
  for(const ref of data.references){
    const fig=document.createElement('figure'),img=document.createElement('img'),caption=document.createElement('figcaption');
    img.src=`${base}references/${ref.processed_photo??String(ref.frame).padStart(6,'0')+'.jpg'}`;img.alt=`実写 フレーム${ref.frame}`;img.loading='lazy';caption.textContent=`実写・フレーム ${ref.frame}`;fig.append(img,caption);
    if(ref.before_overlay){const before=document.createElement('img'),note=document.createElement('figcaption');before.src=`${base}references/${ref.before_overlay}`;before.alt=`修正前 フレーム${ref.frame}`;before.loading='lazy';note.textContent='修正前の配置';fig.append(before,note);}
    if(ref.overlay){const overlay=document.createElement('img'),note=document.createElement('figcaption');overlay.src=`${base}references/${ref.overlay}`;overlay.alt=`推定位置の重ね合わせ フレーム${ref.frame}`;overlay.loading='lazy';note.textContent=ref.before_overlay?'修正後の配置（輪郭は遮蔽処理なし）':'推定の輪郭（遮蔽処理なし・生成画像ではありません）';fig.append(overlay,note);}
    $('refs').append(fig);
  }
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));$('stage').prepend(renderer.domElement);
  const scene=new THREE.Scene();scene.background=new THREE.Color(0x0b121b);
  const camera=new THREE.PerspectiveCamera(50,1,.01,200);camera.up.set(0,0,1);
  const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
  const geometry=await new PLYLoader().loadAsync(`${base}input.ply`);geometry.computeBoundingBox();
  const pointMaterial=new THREE.PointsMaterial({vertexColors:true,size:2,sizeAttenuation:false});
  const cloud=new THREE.Points(geometry,pointMaterial);scene.add(cloud);
  const layout=new THREE.Group(),baseline=new THREE.Group();scene.add(layout,baseline);baseline.visible=false;
  for(const box of data.prediction?.boxes??[]){
    if(!box.evidence.valid)continue;
    const geom=new THREE.BoxGeometry(...box.scale.map(s=>Math.max(s,.025))),col=colorFor(box.label);
    const mesh=new THREE.Mesh(geom,new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:.13,depthWrite:false}));
    mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geom),new THREE.LineBasicMaterial({color:col})));
    const r=box.rotation;mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(r[0][0],r[0][1],r[0][2],0,r[1][0],r[1][1],r[1][2],0,r[2][0],r[2][1],r[2][2],0,0,0,0,1));mesh.position.fromArray(box.center);layout.add(mesh);
  }
  for(const comp of data.baseline){
    const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.Float32BufferAttribute(comp.vertices.flat(),3));geom.setIndex(comp.triangles.flat());
    baseline.add(new THREE.LineSegments(new THREE.EdgesGeometry(geom),new THREE.LineBasicMaterial({color:0xaaaaaa,transparent:true,opacity:.65})));
  }
  const center=geometry.boundingBox.getCenter(new THREE.Vector3()),size=geometry.boundingBox.getSize(new THREE.Vector3()),radius=size.length();
  const grid=new THREE.GridHelper(Math.ceil(radius*1.5),Math.ceil(radius*1.5),0x40546b,0x243347);grid.rotation.x=Math.PI/2;grid.position.set(center.x,center.y,0);scene.add(grid);
  function overview(){camera.position.copy(center).add(new THREE.Vector3(radius*.65,-radius*.8,radius*.65));controls.target.copy(center);controls.update();}
  overview();
  $('overview').onclick=overview;
  $('top').onclick=()=>{camera.position.set(center.x,center.y-.001,center.z+radius*1.25);controls.target.copy(center);controls.update();};
  function update(){cloud.visible=$('points').checked;layout.visible=$('layout').checked;baseline.visible=$('baseline').checked;}
  for(const id of ['points','layout','baseline'])$(id).onchange=update;
  $('pointsOnly').onclick=()=>{$('points').checked=true;$('layout').checked=false;$('baseline').checked=false;update();};
  $('layoutOnly').onclick=()=>{$('points').checked=false;$('layout').checked=true;$('baseline').checked=false;update();};
  $('pointSize').oninput=e=>{pointMaterial.size=Number(e.target.value);};
  const resize=new ResizeObserver(()=>{const w=$('stage').clientWidth,h=$('stage').clientHeight;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);});resize.observe($('stage'));
  renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
}catch(e){$('status').textContent='表示できませんでした';$('error').textContent=e.message;console.error(e);}
