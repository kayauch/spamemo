import * as THREE from 'three';
import {icon} from './spamemo-shell.mjs';
import {findRoomRoute,finitePosition} from './room-route.mjs';

export function installRoomFinder({data,scene,camera,orbit,navigation,walk,startWalking,stopWalking}){
  const $=id=>document.getElementById(id),sceneId=data.transform?.scene_id,rows=data.transform?.generation_world_to_vg_world;
  const validTransform=rows?.length===4&&rows.every(r=>Array.isArray(r)&&r.length===4&&r.every(Number.isFinite));
  const toWorld=validTransform?new THREE.Matrix4().set(...rows.flat()):null;
  const toRoom=toWorld&&Math.abs(toWorld.determinant())>1e-10?toWorld.clone().invert():null;
  const ready=Boolean(sceneId&&toRoom);let target=null,targetPoint=null,route=null,requestVersion=0,abort=null,lastUpdate=0,state='idle';
  const highlight=new THREE.Group(),arrows=new THREE.Group();scene.add(highlight,arrows);
  const key='spamemo-saved-places-'+sceneId;
  function readSaved(){try{const items=JSON.parse(localStorage.getItem(key)||'[]');return Array.isArray(items)?items.map(e=>Array.isArray(data.current_entities)?data.current_entities.find(current=>current.id===e.id):e).filter(e=>e?.scene_id===sceneId&&typeof e.name==='string'&&finitePosition(e.position)):[];}catch{return [];}}
  function savedList(){const items=readSaved();$('savedPlaces').replaceChildren();if(!items.length)$('savedPlaces').textContent='保存した場所はまだありません。';for(const item of items){const b=document.createElement('button');b.type='button';b.textContent=item.name;b.onclick=()=>{$('viewerSettings').close();showTarget(item);};$('savedPlaces').append(b);}}
  function clearGroup(group){for(const mesh of [...group.children]){mesh.geometry?.dispose();mesh.material?.dispose();group.remove(mesh);}}
  function notice(message,registration=false){const root=$('finderNotice');root.replaceChildren(document.createTextNode(message));root.hidden=!message;if(registration&&sceneId){const a=document.createElement('a');a.href='/?scene='+encodeURIComponent(sceneId)+'#memory';a.textContent='ものの位置を登録する';root.append(document.createElement('br'),a);}}
  function setState(value){state=value;document.body.dataset.finderState=value;$('routeBanner').hidden=value!=='guiding';$('roomSearch').hidden=value==='guiding';$('searchSheet').hidden=['idle','loading','guiding'].includes(value);$('goToResult').hidden=value!=='result';$('foundResult').hidden=value!=='arrived';$('savePlace').hidden=value!=='found';$('saveNote').hidden=true;}
  function clear(){++requestVersion;abort?.abort();target=null;targetPoint=null;route=null;clearGroup(highlight);clearGroup(arrows);notice('');setState('idle');$('searchChoices').hidden=true;if(walk.active)stopWalking();}
  function markTarget(point){
    clearGroup(highlight);const ring=new THREE.Mesh(new THREE.TorusGeometry(.1,.012,8,36),new THREE.MeshBasicMaterial({color:0x319fff,depthTest:false,transparent:true,opacity:.8}));ring.position.copy(point);ring.renderOrder=10;highlight.add(ring);
    const pin=new THREE.Mesh(new THREE.SphereGeometry(.025,12,12),new THREE.MeshBasicMaterial({color:0x6db9ff,depthTest:false}));pin.position.copy(point);pin.renderOrder=11;highlight.add(pin);
    if(finitePosition(target?.bbox?.center)&&finitePosition(target?.bbox?.size)){
      const b=target.bbox,size=b.size;if(size.x>0&&size.y>0&&size.z>0){const box=new THREE.Mesh(new THREE.BoxGeometry(size.x,size.y,size.z),new THREE.MeshBasicMaterial({color:0x258bff,transparent:true,opacity:.23,depthWrite:false}));box.position.set(b.center.x,b.center.y,b.center.z);box.updateMatrix();box.applyMatrix4(toRoom);highlight.add(box);}
    }
  }
  function showTarget(entity,answer=''){
    if(!ready||entity.scene_id!==sceneId||!finitePosition(entity.position)){notice('この物の位置を表示できません。');return;}
    target=entity;targetPoint=new THREE.Vector3(entity.position.x,entity.position.y,entity.position.z).applyMatrix4(toRoom);
    if(![targetPoint.x,targetPoint.y,targetPoint.z].every(Number.isFinite)){notice('記録された座標を確認してください。');return;}
    clearGroup(arrows);route=null;stopWalking();markTarget(targetPoint);setState('result');notice('');
    $('resultName').textContent=entity.name;$('resultEyebrow').textContent='見つかった記憶';$('savePlace').querySelector('span').textContent='位置を保存';
    $('resultLocation').textContent=answer||entity.description||'記録された位置';$('resultThumb').replaceChildren(icon('box'));
    const index=entity.metadata?.frame_indices?.find(i=>Number.isInteger(i)&&i>=0);
    if(index!==undefined){const img=document.createElement('img');img.alt='この物を記録したときの写真';img.src=`/scenes/${encodeURIComponent(sceneId)}/frames/${index}`;img.onerror=()=>img.replaceWith(icon('box'));$('resultThumb').replaceChildren(img);}
    const photo=entity.metadata?.shelf_photo;if(typeof photo==='string'&&/^data:image\/(jpeg|png|webp);base64,/.test(photo)){const img=document.createElement('img');img.src=photo;img.alt=entity.name;$('resultThumb').replaceChildren(img);}
    $('goToResult').disabled=!navigation;
    if(!navigation)notice('この部屋には移動範囲の情報がありません。位置の目印を表示します。');
    orbit.target.copy(targetPoint);orbit.update();
  }
  async function search(event){
    event.preventDefault();const text=$('findQuery').value.trim();if(!text||!ready)return;
    clear();const version=++requestVersion;abort=new AbortController();setState('loading');notice('保存した物を探しています…');$('clearSearch').hidden=false;
    try{
      const response=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scene_id:sceneId,text,limit:5}),signal:abort.signal});
      if(!response.ok)throw Error('検索できませんでした。少し待ってから、もう一度お試しください。');const payload=await response.json();if(version!==requestVersion)return;
      const result=payload.query,details=payload.candidate_details||{};
      const hits=(result.candidates?.length?result.candidates:result.hits||[]).map(h=>h.entity).filter(e=>e?.scene_id===sceneId&&finitePosition(e.position));
      const fallback=result.search_mode==='lexical_fallback'?'（意味検索が利用できないため、名前・メモで検索しました）':'';
      if(!hits.length){setState('idle');notice(payload.answer+fallback,true);return;}
      if(result.target){showTarget(result.target,payload.answer);if(fallback)notice(fallback);}
      else{setState('ambiguous');$('resultName').textContent='どれを探していますか？';$('resultEyebrow').textContent='候補を選んでください';$('resultLocation').textContent=payload.answer;$('resultThumb').replaceChildren(icon('search'));notice(fallback);}
      $('searchChoices').replaceChildren();$('searchChoices').hidden=hits.length<2;
      for(const entity of hits){
        const button=document.createElement('button');button.type='button';
        const photo=entity.metadata?.shelf_photo;
        if(typeof photo==='string'&&/^data:image\/(jpeg|png|webp);base64,/.test(photo)){const image=document.createElement('img');image.src=photo;image.alt='';button.append(image);}
        const label=document.createElement('span');label.textContent=entity.name;
        const location=document.createElement('small');location.textContent=details[entity.id]?.location||entity.description||'記録された位置';label.append(location);button.append(label);
        button.setAttribute('aria-pressed',String(entity.id===target?.id));
        button.onclick=()=>{showTarget(entity,details[entity.id]?.answer||'');for(const b of $('searchChoices').children)b.setAttribute('aria-pressed',String(b===button));};$('searchChoices').append(button);
      }
    }catch(error){if(version!==requestVersion||error.name==='AbortError')return;setState('idle');notice(error.message);}
  }
  function drawArrows(points){
    clearGroup(arrows);const shape=new THREE.Shape();shape.moveTo(-.16,-.08);shape.lineTo(0,.12);shape.lineTo(.16,-.08);shape.lineTo(.16,-.16);shape.lineTo(0,.03);shape.lineTo(-.16,-.16);shape.closePath();
    for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i],length=Math.hypot(b.x-a.x,b.y-a.y);for(let d=.15;d<length;d+=.45){const arrow=new THREE.Mesh(new THREE.ShapeGeometry(shape),new THREE.MeshBasicMaterial({color:0x268eff,side:THREE.DoubleSide,transparent:true,opacity:.95}));arrow.position.set(a.x+(b.x-a.x)*d/length,a.y+(b.y-a.y)*d/length,navigation.floor+.05);arrow.rotation.z=Math.atan2(b.y-a.y,b.x-a.x)-Math.PI/2;arrows.add(arrow);}}
  }
  function beginGuide(){
    if(!targetPoint||!navigation)return;
    startWalking();route=findRoomRoute(navigation,camera.position,targetPoint);
    if(!route){stopWalking();notice('この位置まで移動できる経路を確認できませんでした。マップの目印を確認してください。');return;}
    drawArrows(route);setState('guiding');$('routeHeading').textContent=target.name+'の近くへ';$('routeDetail').textContent='青い矢印に沿って進んでください';notice('');
    const next=route.find(p=>Math.hypot(p.x-camera.position.x,p.y-camera.position.y)>.3)||targetPoint;
    const forward=new THREE.Vector3(next.x-camera.position.x,next.y-camera.position.y,0).normalize();
    camera.lookAt(camera.position.x+forward.x,camera.position.y+forward.y,camera.position.z-.3);walk.syncTarget();
  }
  $('roomSearch').onsubmit=search;$('findQuery').disabled=!ready;$('roomSearch').querySelector('[type=submit]').disabled=!ready;
  $('findQuery').addEventListener('input',()=>{const value=$('findQuery').value;clear();$('clearSearch').hidden=!value;});
  $('clearSearch').onclick=()=>{clear();$('findQuery').value='';$('clearSearch').hidden=true;$('findQuery').focus();};$('closeResult').onclick=clear;
  $('goToResult').onclick=beginGuide;$('stopGuide').onclick=()=>{clearGroup(arrows);route=null;stopWalking();setState(target?'result':'idle');};
  $('foundResult').onclick=()=>{if(target?.metadata?.shelf_item)document.dispatchEvent(new CustomEvent('shelf-open-item',{detail:target.id}));clearGroup(arrows);route=null;stopWalking();setState('found');$('resultEyebrow').textContent='見つかりました';};
  $('savePlace').onclick=()=>{if(!target)return;try{const items=readSaved().filter(e=>e.id!==target.id);items.unshift({id:target.id,scene_id:sceneId,name:target.name,position:target.position,description:target.description,metadata:target.metadata});localStorage.setItem(key,JSON.stringify(items.slice(0,100)));$('saveNote').hidden=false;$('saveNote').textContent='このブラウザに位置を保存しました';$('savePlace').querySelector('span').textContent='保存済み';savedList();}catch{$('saveNote').hidden=false;$('saveNote').textContent='位置を保存できませんでした。ブラウザの保存設定をご確認ください。';}};
  for(const id of ['openMap','openSettings','chooseRoom'])$(id).addEventListener('click',()=>{if(walk.active)stopWalking();});
  const canvas=$('roomMap'),ctx=canvas.getContext('2d');
  function drawMap(){
    if(!navigation)return;const xs=navigation.footprint.map(p=>p[0]),ys=navigation.footprint.map(p=>p[1]),lo=[Math.min(...xs),Math.min(...ys)],hi=[Math.max(...xs),Math.max(...ys)];
    const scale=Math.min((canvas.width-70)/(hi[0]-lo[0]),(canvas.height-70)/(hi[1]-lo[1]));
    const ox=(canvas.width-(hi[0]-lo[0])*scale)/2,oy=(canvas.height-(hi[1]-lo[1])*scale)/2;
    const p=(x,y)=>[ox+(x-lo[0])*scale,canvas.height-oy-(y-lo[1])*scale];
    ctx.clearRect(0,0,canvas.width,canvas.height);ctx.lineWidth=3;ctx.strokeStyle='#9bbad1';ctx.fillStyle='#1a3348';ctx.beginPath();navigation.footprint.forEach(([x,y],i)=>ctx[i?'lineTo':'moveTo'](...p(x,y)));ctx.closePath();ctx.fill();ctx.stroke();
    for(const b of navigation.colliders){const r=b.rotation,c=b.center;ctx.beginPath();[[-1,-1],[1,-1],[1,1],[-1,1]].forEach(([x,y],i)=>{const dx=x*b.scale[0]/2,dy=y*b.scale[1]/2;ctx[i?'lineTo':'moveTo'](...p(c[0]+r[0][0]*dx+r[0][1]*dy,c[1]+r[1][0]*dx+r[1][1]*dy));});ctx.closePath();ctx.fillStyle='#3c566d';ctx.fill();}
    if(route){ctx.beginPath();route.forEach((point,i)=>ctx[i?'lineTo':'moveTo'](...p(point.x,point.y)));ctx.strokeStyle='#258fff';ctx.lineWidth=4;ctx.stroke();}
    if(targetPoint){const [x,y]=p(targetPoint.x,targetPoint.y);ctx.fillStyle='#8dcbff';ctx.beginPath();ctx.moveTo(x,y-10);ctx.lineTo(x+9,y);ctx.lineTo(x,y+10);ctx.lineTo(x-9,y);ctx.closePath();ctx.fill();}
    const [x,y]=p(camera.position.x,camera.position.y);ctx.fillStyle='#278dff';ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#c1e3ff';ctx.lineWidth=2;ctx.stroke();
    const forward=camera.getWorldDirection(new THREE.Vector3());ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+forward.x*20,y-forward.y*20);ctx.stroke();
  }
  $('openMap').disabled=!navigation;$('openMap').addEventListener('click',drawMap);savedList();
  if(!ready)notice('このモデルでは検索用の座標を読み込めませんでした。部屋の表示は利用できます。');
  function tick(now){
    if(now-lastUpdate<150)return;lastUpdate=now;
    if($('mapDialog').open)drawMap();
    if(state==='guiding'&&route){const end=route.at(-1),distance=Math.hypot(camera.position.x-end.x,camera.position.y-end.y);if(distance<.4){stopWalking();setState('arrived');$('resultEyebrow').textContent='記録した場所の近くです';notice('周囲を見回して、探している物を確認してください。');}}
    if(highlight.children.length)highlight.children[0].lookAt(camera.position);
  }
  window.addEventListener('pagehide',()=>{abort?.abort();clearGroup(highlight);clearGroup(arrows);},{once:true});
  return {tick,showTarget,beginGuide,drawMap};
}
