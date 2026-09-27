import * as THREE from 'three';
import {installLiveRoomCamera} from './live-room-camera.mjs?v=repair-capture2';
import {captureTargets,targetPhotoUrl} from './capture-targets.mjs?v=20260927-objects1';

import {repairCapturePlan} from './repair-capture-plan.mjs?v=repair-capture2';

export function installRoomCaptureGuide({data,base,run,result,scene,camera,orbit,renderer,surfaces,objectMeshes,stopWalking,startWalking=()=>{},wallPhotos,getViewMode=()=>'overview',setViewMode=()=>{}}){
  const $=id=>document.getElementById(id),sceneId=data.transform?.scene_id;if(!sceneId)return;
  const panel=$('captureGuide'),highlight=new THREE.Group();scene.add(highlight);
  const direct=document.createElement('button');direct.type='button';direct.textContent='保存した写真を壁に合わせる';direct.hidden=true;$('captureThis').after(direct);direct.onclick=()=>{if(selected)wallPhotos?.open(selected.key);};
  const wallChoices=document.createElement('div');wallChoices.className='wall-choices';wallChoices.setAttribute('aria-label','壁を選ぶ');$('captureTarget').after(direct,wallChoices);direct.className='room-primary';direct.style.cssText='width:100%;min-height:44px;margin-top:8px';
  let active=false,targets=[],candidates=[],selected=null,saved=null,down=null,loadPromise=null;
  const objects=new Map([...objectMeshes].map(([key,value])=>[String(key),value]));
  const lookup=new Map();
  const roomBounds=new THREE.Box3();for(const mesh of surfaces.filter(m=>['floor','wall'].includes(m.userData.kind)))roomBounds.expandByObject(mesh);
  function framing(){
    if(!active)return;
    const width=renderer.domElement.clientWidth,height=renderer.domElement.clientHeight;
    const landscape=matchMedia('(max-height:480px) and (orientation:landscape)').matches;
    const covered=landscape?0:Math.min(panel.offsetHeight+24,height*.5),side=landscape?Math.min(panel.offsetWidth+24,width*.6):0;
    camera.setViewOffset(width,height,side/2,covered/2,width,height);
    if(roomBounds.isEmpty())return;
    const center=roomBounds.getCenter(new THREE.Vector3()),direction=new THREE.Vector3(.65,-.8,.85).normalize();
    const right=new THREE.Vector3().crossVectors(camera.up,direction).normalize(),up=new THREE.Vector3().crossVectors(direction,right).normalize();
    const tan=Math.tan(THREE.MathUtils.degToRad(camera.fov/2)),tx=tan*(width-side)/height,ty=tan*(height-covered)/height;
    let distance=1;
    for(const x of [roomBounds.min.x,roomBounds.max.x])for(const y of [roomBounds.min.y,roomBounds.max.y])for(const z of [roomBounds.min.z,roomBounds.max.z]){
      const point=new THREE.Vector3(x,y,z).sub(center),depth=point.dot(direction);
      distance=Math.max(distance,Math.abs(point.dot(right))/tx+depth,Math.abs(point.dot(up))/ty+depth);
    }
    camera.position.copy(center).addScaledVector(direction,distance*1.08);orbit.target.copy(center);orbit.update();
  }
  for(const mesh of surfaces)lookup.set(mesh,mesh.userData.object_id!=null?'object:'+mesh.userData.object_id:'surface:'+mesh.userData.id);
  for(const [key,item] of objects)for(const root of item.meshes)root.traverse(mesh=>{if(mesh.isMesh)lookup.set(mesh,'object:'+key);});
  function clearHighlight(){for(const child of [...highlight.children]){highlight.remove(child);child.material.dispose();if(child.isLineSegments||child.userData.ownedGeometry)child.geometry.dispose();}}
  function meshesFor(target){return target.kind==='object'?(objects.get(target.object_id)?.meshes||[]):surfaces.filter(m=>target.surface_ids.includes(m.userData.id));}
  function select(key,focus=false,shot=null){
    selected=targets.find(t=>t.key===key);if(!selected)return;
    direct.hidden=selected.kind!=='wall';
    for(const button of wallChoices.children)button.setAttribute('aria-pressed',String(button.dataset.key===selected.key));
    $('captureTarget').value=selected.key;$('captureTargetTitle').textContent=selected.repair?selected.title+'を確認':selected.title+'を撮ってください';
    $('captureInstruction').textContent=selected.instruction;$('captureTargetNote').textContent=selected.note;
    $('captureTargetNote').dataset.repair=String(selected.repair);
    const index=candidates.indexOf(selected);$('captureCandidateCount').textContent=index>=0?`撮影・確認箇所 ${index+1} / ${candidates.length}`:'自分で選んだ対象';
    $('capturePrev').disabled=candidates.length<2;$('captureNext').disabled=!candidates.length;
    $('captureThis').href=targetPhotoUrl(sceneId,run,result,selected.key);$('captureThis').hidden=false;
    $('captureReference').hidden=!selected.frames.length;
    $('captureReference').open=false;
    if(selected.frames.length){$('captureReferenceImage').src='/scenes/'+encodeURIComponent(sceneId)+'/frames/'+selected.frames[0];}
    else $('captureReferenceImage').removeAttribute('src');
    clearHighlight();scene.updateMatrixWorld(true);
    const seen=new Set(),box=new THREE.Box3();
    for(const root of meshesFor(selected))root.traverse(mesh=>{
      if(!mesh.isMesh||seen.has(mesh))return;seen.add(mesh);
      const overlay=new THREE.Mesh(mesh.geometry,new THREE.MeshBasicMaterial({color:0x168fff,transparent:true,opacity:.38,side:THREE.DoubleSide,depthTest:false,depthWrite:false}));
      overlay.matrixAutoUpdate=false;overlay.matrix.copy(mesh.matrixWorld);overlay.renderOrder=20;highlight.add(overlay);
      const edge=new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),new THREE.LineBasicMaterial({color:0x8ed4ff,depthTest:false,transparent:true,opacity:.85}));
      edge.matrixAutoUpdate=false;edge.matrix.copy(mesh.matrixWorld);edge.renderOrder=21;highlight.add(edge);box.expandByObject(mesh);
    });
    if(shot?.region&&selected.shape?.origin&&selected.shape?.u&&selected.shape?.v){
      // The shooting cell is a guide on the estimated surface, not camera localization.
      for(const child of highlight.children)child.material.opacity*=.22;
      const {u0,u1,v0,v1}=shot.region,shape=selected.shape;
      const origin=new THREE.Vector3(...shape.origin),u=new THREE.Vector3(...shape.u),v=new THREE.Vector3(...shape.v);
      const points=[[u0,v0],[u1,v0],[u1,v1],[u0,v1]].map(([x,y])=>origin.clone().addScaledVector(u,x).addScaledVector(v,y));
      const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(points.flatMap(p=>p.toArray()),3));geometry.setIndex([0,1,2,0,2,3]);
      const patch=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:0x199fff,transparent:true,opacity:.66,side:THREE.DoubleSide,depthTest:false,depthWrite:false}));patch.userData.ownedGeometry=true;patch.renderOrder=23;highlight.add(patch);
      const edge=new THREE.LineSegments(new THREE.EdgesGeometry(geometry),new THREE.LineBasicMaterial({color:0xe3f6ff,depthTest:false}));edge.renderOrder=24;highlight.add(edge);
    }
    if(focus)framing();
  }
  const liveCamera=installLiveRoomCamera({sceneId,run,result,showTarget:(key,shot)=>select(key,true,shot),previewImage:()=>{
    renderer.render(scene,camera);
    const source=renderer.domElement,width=source.clientWidth,height=source.clientHeight,landscape=matchMedia('(max-height:480px) and (orientation:landscape)').matches;
    const contentW=width-(landscape?Math.min(panel.offsetWidth+24,width*.6):0),contentH=height-(landscape?0:Math.min(panel.offsetHeight+24,height*.5));
    const canvas=document.createElement('canvas');canvas.width=320;canvas.height=Math.round(320*contentH/contentW);
    canvas.getContext('2d').drawImage(source,0,0,source.width*contentW/width,source.height*contentH/height,0,0,canvas.width,canvas.height);
    return canvas.toDataURL('image/jpeg',.85);
  }});
  $('captureThis').onclick=event=>{event.preventDefault();if(selected)liveCamera.open(repairCapturePlan(targets,{selectedKey:selected.key}));};
  $('captureThis').textContent='不足箇所を順番に撮影';
  $('captureGuideHint').after($('captureThis'));$('captureThis').style.cssText='width:100%;box-sizing:border-box;margin:10px 0;justify-content:center';
  async function load(){
    let quality={};
    try{const response=await fetch(base+'quality.json');if(!response.ok)throw Error();quality=await response.json();}
    catch{$('captureGuideHint').textContent='品質情報を読み込めません。3D上の面をタップするか、一覧から選んでください。';}
    targets=captureTargets(data,quality);candidates=targets.filter(t=>t.photo||!t.coverageKnown||t.supportedFraction<.9).sort((a,b)=>b.score-a.score);
    const plan=repairCapturePlan(targets);
    $('captureGuideHint').textContent=`青い場所から、壁・床・天井の区画と家具の向きを順に案内します（目安${plan.length}枚）。横向きで撮影してください。写真不足や形のずれは推定です。`;
    $('captureTarget').replaceChildren();for(const target of targets)$('captureTarget').add(new Option(target.title+(target.repair?'（形状も確認）':target.candidate?'（写真不足の候補）':''),target.key));
    wallChoices.replaceChildren();for(const t of targets.filter(t=>t.kind==='wall')){const b=document.createElement('button');b.type='button';b.textContent=t.title;b.dataset.key=t.key;b.onclick=()=>select(t.key,true);wallChoices.append(b);}
    if(!targets.length){$('captureTargetTitle').textContent='選択できる面が見つかりません';return;}
    if(active){const requested=new URLSearchParams(location.search).get('target');select(targets.find(t=>t.key===requested)?.key||candidates[0]?.key||targets[0].key,true);}
  }
  function enter(event){
    event?.preventDefault();if(active)return;stopWalking();$('viewerSettings').close();$('closeResult').click();$('stopGuide').click();
    saved={viewMode:getViewMode(),position:camera.position.clone(),target:orbit.target.clone(),visibility:surfaces.map(m=>m.visible),ceiling:$('ceiling').checked};
    active=true;document.body.dataset.captureGuide='true';panel.hidden=false;
    $('overview').click();
    if(targets.length)select(selected?.key||candidates[0]?.key||targets[0].key,true);else{loadPromise??=load();}
  }
  function exit(){
    if(!active)return;active=false;panel.hidden=true;delete document.body.dataset.captureGuide;clearHighlight();
    camera.clearViewOffset();
    if(saved){setViewMode(saved.viewMode);camera.position.copy(saved.position);orbit.target.copy(saved.target);surfaces.forEach((mesh,i)=>mesh.visible=saved.visibility[i]);$('ceiling').checked=saved.ceiling;orbit.update();if(saved.viewMode==='walk')startWalking();}
    (document.getElementById('exploreRepair')||$('addRoomPhotos')).focus();
  }
  for(const id of ['addRoomPhotos','settingsAddPhotos'])$(id).onclick=enter;
  $('addRoomPhotos').textContent='部屋の修正';
  document.addEventListener('repair-room',enter);
  $('captureGuideClose').onclick=exit;
  $('captureTarget').onchange=()=>select($('captureTarget').value,true);
  function step(delta){if(!candidates.length)return;let index=candidates.indexOf(selected);select(candidates[(index<0?0:(index+delta+candidates.length)%candidates.length)].key,true);}
  $('capturePrev').onclick=()=>step(-1);$('captureNext').onclick=()=>step(1);
  const raycaster=new THREE.Raycaster();
  renderer.domElement.addEventListener('pointerdown',event=>{if(active)down={id:event.pointerId,x:event.clientX,y:event.clientY,multi:!event.isPrimary};});
  renderer.domElement.addEventListener('pointercancel',()=>{down=null;});
  renderer.domElement.addEventListener('pointerup',event=>{
    if(!active||!down||down.multi||down.id!==event.pointerId)return;
    const start=down;down=null;if(Math.hypot(event.clientX-start.x,event.clientY-start.y)>7)return;
    const rect=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1),camera);
    const visible=mesh=>{for(let node=mesh;node;node=node.parent)if(!node.visible)return false;return true;};
    const pickable=[...lookup.keys()].filter(visible);
    const hits=raycaster.intersectObjects(selected?.kind==='wall'?pickable.filter(m=>m.userData.kind==='wall'):pickable,false);
    if(hits[0])select(lookup.get(hits[0].object));
  });
  document.addEventListener('keydown',event=>{if(active&&event.key==='Escape'&&!document.querySelector('dialog[open]'))exit();});
  new ResizeObserver(()=>{if(active)framing();}).observe(renderer.domElement);
  document.addEventListener('select-room-wall',async()=>{enter();if(loadPromise)await loadPromise;const wall=targets.find(t=>t.kind==='wall');if(wall)select(wall.key,true);});
  if(new URLSearchParams(location.search).get('capture')==='1')enter();
}
