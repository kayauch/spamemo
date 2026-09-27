import {loadObjectCorrections} from './object-corrections.mjs?v=20260927-objects1';
import {installObjectEditor} from './object-editor.mjs?v=20260927-objects1';
import {createStorageHighlight} from './storage-highlight.mjs?v=1';
import {installRoomReview} from './room-review.mjs?v=20260927-objects1';
import {openingPreview} from './room-opening.mjs?v=1';
import {installWallPhotos} from './wall-photo.mjs?v=1';
import {installExplorer} from './explorer.mjs?v=repair-capture2';
import {installMinimalRoom} from './minimal-room.mjs?v=1';
import {installShelves} from './shelves.mjs?v=20260927-objects1';
import {installRoomCaptureGuide} from './room-capture-guide.mjs?v=20260927-objects1';
import {installRoomFinder} from './room-finder.mjs?v=20260927-objects1';
import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {WalkControls} from './walk-controls.js';
import {RoomNavigation} from './room-navigation.js';
import {loadSurfaceTextures} from './surface-texture-loader.mjs';
const $=id=>document.getElementById(id);
let opening;
try {
  const query=new URLSearchParams(location.search),run=query.get('run'),result=query.get('result');
  if(!run||!result)throw Error('表示する部屋が指定されていません。ホームで作成済みの部屋を選んでください。');
  if(![run,result].every(v=>/^[a-zA-Z0-9_-]+$/.test(v)))throw Error('結果名が不正です');
  const base=`/static/generation-reviews/${run}/${result}/`,response=await fetch(base+'room.json');
  if(!response.ok)throw Error('部屋の写真処理がまだ完了していません');
  const corrections=await loadObjectCorrections(await response.json());
  const data=corrections.data;
  opening=openingPreview(data);
  if(data.photo_extension?.enabled){const note=document.createElement('p');note.className='dialog-note';note.textContent='壁・床・天井には、同じ元写真の続きを広めに貼っています。延長した部分の位置は概算です。写真の範囲確認では、位置の一致を厳しく確認できた部分だけを表示します。';$('walkNote').after(note);}
  if(data.pattern_completion?.enabled){const note=document.createElement('p');note.className='dialog-note';note.textContent='壁・天井の空白は、写真から切り出した模様を繰り返して補っています。補った模様は実際の撮影結果ではありません。';$('walkNote').after(note);}
  if(data.color_completion?.enabled){const note=document.createElement('p');note.className='dialog-note';note.textContent='写真がない部分は周囲の色で補っています。「写真を貼れた範囲を確認」で実際の撮影範囲を見られます。';$('walkNote').after(note);}
  // Use the model's source scene for memory/search, not an unrelated query value.
  const sourceScene=data.transform?.scene_id;
  if(sourceScene){
    const alignment=$('photoAlignment');if(alignment){alignment.href=`/static/photo-alignment.html?run=${run}&result=${result}`;alignment.hidden=false;if(data.photo_alignment)alignment.textContent='写真との位置合わせを確認';}
    const addPhotos=$('addRoomPhotos');if(addPhotos){addPhotos.href='/static/add-room-photos.html?scene='+encodeURIComponent(sourceScene);addPhotos.hidden=false;}
    const settingsPhotos=$('settingsAddPhotos');if(settingsPhotos){settingsPhotos.href='/static/add-room-photos.html?scene='+encodeURIComponent(sourceScene);settingsPhotos.hidden=false;}
    const improve=$('captureImprove');if(improve){improve.href='/room/new?scene='+encodeURIComponent(sourceScene);improve.hidden=false;}
    const scenePath='/?scene='+encodeURIComponent(sourceScene);
    for(const [id,screen] of [['registerLink','memory'],['searchLink','find']]){
      $(id).href=scenePath+'#'+screen;$(id).hidden=false;
    }
    fetch('/viewer/state/'+encodeURIComponent(sourceScene)).then(r=>r.ok?r.json():null).then(state=>{
      if(state?.scene?.name){$('roomTitle').textContent=state.scene.name;$('openingTitle').textContent=state.scene.name+'を開いています';document.title=state.scene.name+' | Spatial Memory';}
    }).catch(()=>{});
  }
  if(data.model_url){$('download').href=base+data.model_url;$('download').hidden=false;}
  $('compare').href=`/static/generation-review.html?run=${run}&prediction=${data.source_prediction}`;
  const scene=new THREE.Scene();scene.background=new THREE.Color('#222b30');
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;$('stage').prepend(renderer.domElement);
  const camera=new THREE.PerspectiveCamera(65,1,.025,100);camera.up.set(0,0,1);
  const orbit=new OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;
  let viewMode='overview';
  const surfaces=[],loader=new THREE.TextureLoader();
  const objectMeshes=new Map();
  function registerObject(id,label,mesh){
    if(!objectMeshes.has(id))objectMeshes.set(id,{label,name:data.object_names?.[String(id)],meshes:[]});
    objectMeshes.get(id).meshes.push(mesh);
  }
  // Scene-specific experimental appearance assets are not distributed in this snapshot.
  let aiTexture=null;const roomTextures={};
  const floorGroup=new THREE.Group();scene.add(floorGroup);
  const surfaceTextures=await loadSurfaceTextures(data.surfaces,async path=>{const texture=await loader.loadAsync(base+path);opening.photo(path,texture);return texture;},{
    concurrency:8,onProgress:(completed,total)=>{$('status').textContent=`写真を壁に戻しています（${completed} / ${total}）`;opening.progress(completed,total);}
  });
  for(const [surfaceIndex,s] of data.surfaces.entries()){
    const origin=new THREE.Vector3(...s.origin),u=new THREE.Vector3(...s.u),v=new THREE.Vector3(...s.v);
    let positions,uvs,indices;
    if(s.polygon){
      const shape=s.polygon.map(p=>new THREE.Vector2(...p));
      positions=shape.flatMap(p=>[p.x,p.y,s.origin[2]]);
      uvs=shape.flatMap(p=>[(p.x-origin.x)/u.x,(p.y-origin.y)/v.y]);
      indices=THREE.ShapeUtils.triangulateShape(shape,[]).flat();
    }else{
      positions=[origin,origin.clone().add(u),origin.clone().add(u).add(v),origin.clone().add(v)].flatMap(p=>p.toArray());
      uvs=[0,0,1,0,1,1,0,1];indices=[0,1,2,0,2,3];
    }
    const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geom.setAttribute('uv',new THREE.Float32BufferAttribute(uvs,2));geom.setIndex(indices);geom.computeVertexNormals();
    const {texture,mask}=surfaceTextures[surfaceIndex];texture.colorSpace=THREE.SRGBColorSpace;texture.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());
    const generated=s.id==='cabinet-3000-1-1'?aiTexture:null;
    let roomGenerated=null;
    if(roomTextures.wall){
      if(s.id==='wall-11')roomGenerated=roomTextures['window-wall'];
      else if(s.kind==='ceiling'||['wall-0','wall-1','wall-8'].includes(s.id)){
        roomGenerated=roomTextures[s.kind==='ceiling'?'ceiling':'wall'].clone();
        roomGenerated.wrapS=roomGenerated.wrapT=THREE.RepeatWrapping;
        const tileMetres=s.kind==='ceiling'?1.8:1;
        roomGenerated.repeat.set(u.length()/tileMetres,v.length()/tileMetres);roomGenerated.needsUpdate=true;
      }
    }
    const material=new THREE.MeshBasicMaterial({map:roomGenerated??generated??texture,side:THREE.DoubleSide});
    const mesh=new THREE.Mesh(geom,material);mesh.userData={...s,texture,mask,generated,roomGenerated};scene.add(mesh);surfaces.push(mesh);
    if(s.kind==='floor')floorGroup.add(mesh);
    if(s.object_id!=null)registerObject(s.object_id,s.object_label,mesh);
  }
  // Furniture details are reusable approximations inside the AI-estimated boxes.
  scene.add(new THREE.AmbientLight(0xffffff,2));
  const light=new THREE.DirectionalLight(0xfff9e9,2);light.position.set(2,3,8);scene.add(light);
  const material=color=>new THREE.MeshStandardMaterial({color,roughness:.85});
  const wood=material('#987146'),metal=material('#343638'),red=material('#7e2836'),seat=material('#292829');
  function part(group,scale,center,mat){const mesh=new THREE.Mesh(new THREE.BoxGeometry(...scale),mat);mesh.position.set(...center);group.add(mesh);return mesh;}
  for(const b of data.furniture){
    const group=new THREE.Group(),[w,d,h]=b.scale,r=b.rotation;
    group.position.fromArray(b.center);group.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(r[0][0],r[0][1],r[0][2],0,r[1][0],r[1][1],r[1][2],0,r[2][0],r[2][1],r[2][2],0,0,0,0,1));scene.add(group);
    registerObject(b.id,b.label,group);
    if(['chair','dining_chair'].includes(b.label)){
      const office=b.label==='chair',finish=office?seat:wood;
      part(group,[w*.82,d*.8,h*.07],[0,0,-h*.04],finish);
      part(group,[w*.8,d*.09,h*.46],[0,d*.36,h*.25],office?red:wood);
      if(office){
        part(group,[w*.08,d*.08,h*.4],[0,0,-h*.26],metal);
        for(let i=0;i<5;i++){const angle=i*Math.PI*2/5;const leg=part(group,[w*.43,d*.07,h*.055],[Math.cos(angle)*w*.2,Math.sin(angle)*w*.2,-h*.46],metal);leg.rotation.z=angle;}
      }else for(const x of [-1,1])for(const y of [-1,1])part(group,[w*.065,d*.065,h*.43],[x*w*.34,y*d*.3,-h*.28],metal);
    }else if(['illumination','chandelier'].includes(b.label)){
      group.position.z=Math.min(group.position.z,data.ceiling-h/2);
      part(group,[w,d,h],[0,0,0],new THREE.MeshBasicMaterial({color:'#f9f5df'}));
    }else part(group,b.scale,[0,0,0],wood);
  }
  const bounds=new THREE.Box3().setFromObject(floorGroup),center=bounds.getCenter(new THREE.Vector3()),diagonal=bounds.getSize(new THREE.Vector3()).length();
  // The largest triangle gives a guaranteed interior location even for a concave room.
  const floorGeom=floorGroup.children[0].geometry,pos=floorGeom.attributes.position,idx=floorGeom.index;
  let area=-1,spawn=center.clone();
  for(let i=0;i<idx.count;i+=3){const a=new THREE.Vector3().fromBufferAttribute(pos,idx.getX(i)),b=new THREE.Vector3().fromBufferAttribute(pos,idx.getX(i+1)),c=new THREE.Vector3().fromBufferAttribute(pos,idx.getX(i+2));const size=new THREE.Vector3().crossVectors(b.clone().sub(a),c.clone().sub(a)).length();if(size>area){area=size;spawn.copy(a).add(b).add(c).divideScalar(3);}}
  function walkingUI(active){
    $('walk').setAttribute('aria-pressed',String(active));
    for(const button of document.querySelectorAll('[data-walk-key]'))button.disabled=!navigation||!safeSpawn;
  }
  function stop(){walk.stop();walkingUI(false);$('walk').textContent='中を歩く';$('hint').textContent='ドラッグ：見回す ／ ホイール：拡大';}
  const walk=new WalkControls(camera,renderer.domElement,()=>orbit,stop);
  const navigation=data.navigation?new RoomNavigation(data.navigation):null;
  const safeSpawn=navigation?.findSpawn(data.navigation.spawn?.slice(0,2)||[spawn.x,spawn.y]);
  if(navigation){
    if(safeSpawn)spawn.set(safeSpawn.x,safeSpawn.y,safeSpawn.z);
    else{$('walk').disabled=true;$('error').textContent='家具を避けて立てる場所が見つかりません。推定枠を確認してください。';}
    walk.constrainMove=(position,motion)=>navigation.move(position,motion);
  }
  $('walkNote').textContent=navigation?'歩行中は床の高さを保ち、壁と家具を通り抜けません。入口は閉じた面として扱います。':'以前の結果です。歩行中は壁を通り抜けられます。作り直すと衝突判定付きになります。';
  function ceiling(){surfaces.filter(m=>m.userData.kind==='ceiling').forEach(m=>m.visible=$('ceiling').checked);}
  function walls(visible){surfaces.filter(m=>m.userData.kind==='wall').forEach(m=>m.visible=visible);}
  function inside(){
    viewMode='walk';stop();walls(true);camera.position.set(spawn.x,spawn.y,data.floor+Math.min(1.55,(data.ceiling-data.floor)*.65));
    let focal=null,volume=0;
    for(const item of objectMeshes.values()){
      if(['curtain','illumination','chandelier'].includes(item.label))continue;
      const box=new THREE.Box3();for(const mesh of item.meshes)box.expandByObject(mesh);
      const size=box.getSize(new THREE.Vector3()),point=box.getCenter(new THREE.Vector3()),v=size.x*size.y*size.z;
      if(v>volume&&Math.hypot(point.x-spawn.x,point.y-spawn.y)>.3){volume=v;focal=point;}
    }
    orbit.target.copy(focal||new THREE.Vector3(center.x,bounds.max.y,camera.position.z-.12));
    orbit.update();$('ceiling').checked=true;ceiling();
  }
  $('inside').onclick=inside;
  $('ceilingView').onclick=()=>{inside();orbit.target.set(camera.position.x+.25,camera.position.y+1,data.ceiling+.8);orbit.update();};
  $('aiView').onclick=()=>{
    const mesh=surfaces.find(m=>m.userData.generated);if(!mesh)return;stop();
    const s=mesh.userData,u=new THREE.Vector3(...s.u),v=new THREE.Vector3(...s.v),target=new THREE.Vector3(...s.origin).addScaledVector(u,.5).addScaledVector(v,.5);
    const normal=new THREE.Vector3().crossVectors(u,v).normalize();if(normal.dot(center.clone().sub(target))<0)normal.negate();
    camera.position.copy(target).addScaledVector(normal,2.4).add(new THREE.Vector3(0,0,.35));orbit.target.copy(target);orbit.update();
  };
  $('overview').onclick=()=>{viewMode='overview';stop();walls(true);camera.position.copy(center).add(new THREE.Vector3(diagonal*.65,-diagonal*.8,diagonal*.85));orbit.target.copy(center).add(new THREE.Vector3(0,0,1));orbit.update();$('ceiling').checked=false;ceiling();};
  $('furnitureView').onclick=()=>{$('overview').onclick();walls(false);camera.position.copy(center).add(new THREE.Vector3(diagonal*.4,-diagonal*.45,diagonal*.6));orbit.target.copy(center).add(new THREE.Vector3(0,0,.6));orbit.update();};
  $('ceiling').onchange=ceiling;
  if(objectMeshes.size){
    const panel=document.createElement('div'),label=document.createElement('label'),select=document.createElement('select'),note=document.createElement('span');
    panel.className='furniture-picker';select.setAttribute('aria-label','近くで見る家具');
    label.textContent='家具を近くで見る：';label.append(select);panel.append(label,note);$('furnitureView').parentElement.after(panel);
    select.add(new Option('家具を選択',''));
    const names={bed:'ベッド',locker:'ロッカー',shelf:'棚',box:'収納箱',wardrobe:'収納',desk:'机',chair:'椅子',dining_chair:'椅子',curtain:'カーテン',tub:'浴槽（推定）',pillow:'枕',painting:'壁の飾り',coffee_table:'テーブル',cushion:'クッション',wall_decoration:'壁の飾り',chandelier:'照明'};
    for(const [id,item] of objectMeshes)select.add(new Option(item.name||`${names[item.label]||item.label} ${id}`,String(id)));
    select.onchange=()=>{
      const item=objectMeshes.get(Number(select.value));if(!select.value||!item)return;
      viewMode='overview';stop();walls(false);$('ceiling').checked=false;ceiling();
      const box=new THREE.Box3();for(const mesh of item.meshes)box.expandByObject(mesh);
      const target=box.getCenter(new THREE.Vector3()),distance=Math.max(.6,box.getSize(new THREE.Vector3()).length());
      const direction=center.clone().sub(target);direction.z=0;if(direction.length()<.1)direction.set(0,-1,0);direction.normalize();
      camera.position.copy(target).addScaledVector(direction,distance).add(new THREE.Vector3(0,0,distance*.35));orbit.target.copy(target);orbit.update();
      const faces=item.meshes.filter(m=>m.userData.supported_fraction!=null);
      note.textContent=faces.length?' 写真が貼れていない面は無地です。「写真を貼れた範囲」で確認できます。':' 細部は仮の形状です。';
    };
  }
  function appearance(){for(const mesh of surfaces){const d=mesh.userData;mesh.material.map=$('support').checked?d.mask:($('roomAi').checked&&d.roomGenerated?d.roomGenerated:($('ai').checked&&d.generated?d.generated:d.texture));mesh.material.needsUpdate=true;}}
  $('support').onchange=appearance;$('ai').onchange=appearance;$('roomAi').onchange=appearance;
  $('walk').onclick=()=>{if(walk.active)stop();else{
    if(navigation){
      const point=navigation.findSpawn([camera.position.x,camera.position.y]);
      if(!point)return;
      // Overview/furniture inspection may start outside or above the room.
      inside();const direction=camera.getWorldDirection(new THREE.Vector3());camera.position.set(point.x,point.y,point.z);orbit.target.copy(camera.position).add(direction);orbit.update();
    }
    viewMode='walk';walk.start(diagonal);walkingUI(true);if(navigation)walk.speed=1.2;
    $('walk').textContent='歩行を終了';$('hint').textContent=navigation?'WASD / 矢印：歩く ／ ドラッグ：見回す ／ Esc：終了（壁・家具で止まります）':'WASD / 矢印：移動 ／ ドラッグ：見回す ／ Q・E：上下 ／ Esc：終了';
  }};
  new ResizeObserver(()=>{const w=$('stage').clientWidth,h=$('stage').clientHeight;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);}).observe($('stage'));
  inside();
  function startRoomWalking(){
    if(walk.active||!navigation)return;
    const point=navigation.findSpawn([camera.position.x,camera.position.y]);if(!point)return;
    viewMode='walk';
    const direction=camera.getWorldDirection(new THREE.Vector3());direction.z=0;
    if(direction.lengthSq()<1e-8)direction.set(0,1,0);direction.normalize();
    walls(true);$('ceiling').checked=true;ceiling();
    camera.position.set(point.x,point.y,point.z);orbit.target.copy(camera.position).add(direction);orbit.update();
    walk.start(diagonal);walk.speed=1.2;walkingUI(true);$('walk').textContent='歩行を終了';
  }
  const finder=installRoomFinder({data,scene,camera,orbit,navigation,walk,startWalking:startRoomWalking,stopWalking:stop});
  let objectEditor=null;
  const editHighlight=createStorageHighlight(scene);
  function openObjectEditor(id){
    if(!objectEditor)return;
    stop();$('viewerSettings').close();$('closeResult').click();
    if(document.body.dataset.captureGuide==='true')$('captureGuideClose').click();
    objectEditor.open(id);
  }
  if(corrections.available){
    objectEditor=installObjectEditor({objects:()=>corrections.objects,save:corrections.save,clearFocus:()=>editHighlight.clear(),focus:id=>{
      const item=[...objectMeshes].find(([key])=>String(key)===String(id))?.[1];if(!item)return;
      $('furnitureView').onclick();const box=editHighlight.show(item.meshes);if(box.isEmpty())return;
      const target=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
      const direction=center.clone().sub(target);direction.z=0;if(direction.length()<.1)direction.set(0,-1,0);direction.normalize();
      const mobile=renderer.domElement.clientWidth<=700,distance=Math.max(2.8,size.length()*(mobile?2.05:1.7));
      const aim=target.clone();if(mobile)aim.z-=Math.max(.45,size.z*.48);
      camera.position.copy(aim).addScaledVector(direction,distance);camera.position.z+=distance*.32;orbit.target.copy(aim);orbit.update();
    }});
    for(const parent of [$('viewerSettings').querySelector('.settings-links'),$('captureGuide')]){
      const button=document.createElement('button');button.type='button';button.className='room-secondary';button.textContent='認識した家具を修正';button.onclick=()=>openObjectEditor();parent.append(button);
    }
    addEventListener('pagehide',()=>editHighlight.dispose(),{once:true});
  }
  await installShelves({data,scene,camera,orbit,renderer,floorGroup,finder,navigation,objectMeshes,overview:()=>$('furnitureView').onclick(),stopWalking:stop,editObject:corrections.available?openObjectEditor:null});
  // A first press starts walking before the existing pointer handler handles movement.
  for(const button of document.querySelectorAll('[data-walk-key]')){
    button.disabled=!navigation||!safeSpawn;
    button.addEventListener('pointerdown',()=>{startRoomWalking();},{capture:true});
    button.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();startRoomWalking();walk.keys.add(button.dataset.walkKey);}});
    button.addEventListener('keyup',()=>walk.keys.delete(button.dataset.walkKey));
    button.addEventListener('blur',()=>walk.keys.delete(button.dataset.walkKey));
  }
  const wallPhotos=await installWallPhotos({data,run,result,surfaces,stopWalking:stop});
  installMinimalRoom(stop);
  const explorer=installExplorer({camera,orbit,renderer,finder,overview:()=>$('overview').onclick(),startWalking:startRoomWalking,getViewMode:()=>viewMode,canWalk:()=>Boolean(navigation&&safeSpawn)});
  const roomReview=installRoomReview({data,surfaces,objectMeshes,camera,orbit,scene,walk,stopWalking:stop,overview:()=>$('overview').onclick()});
  installRoomCaptureGuide({data,base,run,result,scene,camera,orbit,renderer,surfaces,objectMeshes,stopWalking:stop,startWalking:startRoomWalking,wallPhotos,getViewMode:()=>viewMode,setViewMode:value=>{viewMode=value;}});
  opening?.dispose();document.body.classList.add('viewer-ready');
  renderer.setAnimationLoop(now=>{if(walk.active)walk.update(now);else orbit.update();finder.tick(now);explorer.tick(now);roomReview.tick(now);renderer.render(scene,camera);});
  $('status').textContent=navigation?'実写テクスチャ付きの部屋モデル — 床・壁・天井の閉じた形状を検証済み':'点群なし・実写を貼った面の表示（試作）';
  if(data.image_furniture_summary?.confirmed_count===0)$('status').textContent+=' — 複数画像で確認できた家具はありません（空室または検出漏れ）';
  if(data.vertical_anchors?.ceiling_observed===false)$('status').textContent+=' — 天井は未観測のため仮の高さです';
}catch(error){opening?.dispose();$('openingTitle').textContent='部屋を開けませんでした';document.body.classList.add('viewer-failed');$('status').textContent='表示できませんでした';$('error').textContent=error.message;console.error(error);}
