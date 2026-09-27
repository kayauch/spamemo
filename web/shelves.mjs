import {installStoragePhotos} from './storage-photo.mjs?v=1';
import {createStorageHighlight} from './storage-highlight.mjs?v=1';
import * as THREE from 'three';
export async function installShelves({data,scene,camera,orbit,renderer,floorGroup,finder,overview,stopWalking,navigation,objectMeshes,editObject=null}){
 const sceneId=data.transform?.scene_id;if(!sceneId)return;
 const matrix=new THREE.Matrix4().set(...data.transform.generation_world_to_vg_world.flat());
 const group=new THREE.Group();scene.add(group);let entities=[],placing=null,preview=null,selected=null,highlightedName='';
 const storageHighlight=createStorageHighlight(scene);
 const detected=[];const counts={};const labels={cabinet:'キャビネット',shelf:'棚',locker:'ロッカー',wardrobe:'収納',box:'収納箱',desk:'机'};
 for(const [id,item] of objectMeshes){if(!labels[item.label])continue;counts[item.label]=(counts[item.label]||0)+1;detected.push({id:String(id),item,name:item.name||labels[item.label]+' '+counts[item.label]});}
 const ray=new THREE.Raycaster(),pointer=new THREE.Vector2();
 const style=document.createElement('link');style.rel='stylesheet';style.href='/static/shelves.css?v=3';document.head.append(style);
 const launcher=document.createElement('button');launcher.className='shelf-launch';launcher.textContent='収納';launcher.onclick=library;document.getElementById('stage').append(launcher);
 const dialog=document.createElement('dialog');dialog.className='shelf-dialog';document.body.append(dialog);
 const banner=document.createElement('div');banner.className='shelf-placement';banner.hidden=true;document.getElementById('stage').append(banner);
 function el(tag,text,cls){const e=document.createElement(tag);if(text)e.textContent=text;if(cls)e.className=cls;return e;}
 function button(text,fn,cls=''){const b=el('button',text,cls);b.type='button';b.onclick=fn;return b;}
 function page(title){dialog.replaceChildren();const head=el('header');head.append(el('h2',title),button('閉じる',()=>dialog.close()));dialog.append(head);const status=el('p','','shelf-selection-status');status.setAttribute('role','status');status.setAttribute('aria-live','polite');dialog.append(status);selectionStatus();if(!dialog.open)dialog.showModal();return dialog;}
 function selectionStatus(){dialog.classList.toggle('shelf-preview-dialog',Boolean(highlightedName));const status=dialog.querySelector('.shelf-selection-status');if(status){status.hidden=!highlightedName;status.textContent=highlightedName+' を青く表示しています';}}
 function clearSelection(){storageHighlight.clear();highlightedName='';selectionStatus();for(const b of dialog.querySelectorAll('.shelf-list button'))b.removeAttribute('aria-current');}
 dialog.addEventListener('close',()=>{if(!dialog.open&&!document.querySelector('.storage-photo-dialog[open]'))clearSelection();});
 window.addEventListener('pagehide',()=>storageHighlight.dispose(),{once:true});
 function message(text){let p=dialog.querySelector('.shelf-error');if(!p){p=el('p','','shelf-error');p.setAttribute('role','alert');dialog.append(p);}p.textContent=text;}
 async function request(url,options){const r=await fetch(url,options);if(!r.ok)throw Error('保存・読み込みに失敗しました。もう一度お試しください。');return r.json();}
 async function save(entity){return request('/entities',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(entity)});}
 function field(form,label,value,type='text',min,max){const wrap=el('label',label),input=document.createElement('input');input.type=type;input.value=value;input.required=true;if(min!=null)input.min=min;if(max!=null)input.max=max;if(type==='number')input.step='1';if(type==='text')input.maxLength=120;wrap.append(input);form.append(wrap);return input;}
 function shelves(){return entities.filter(e=>e.metadata?.shelf_ui===1&&(!e.metadata.source_object_id||(e.metadata.source_prediction===data.source_prediction&&detected.some(d=>d.id===String(e.metadata.source_object_id)))));}
 function items(id){return entities.filter(e=>e.parent_id===id&&e.metadata?.shelf_item===1);}
 function dispose(g){g.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});g.removeFromParent();}
 function meshShelf(s,ghost=false){const m=s.metadata, g=new THREE.Group();g.position.fromArray(m.room_position);g.rotation.z=m.angle*Math.PI/180;g.userData.entity=s;
  const mat=new THREE.MeshBasicMaterial({color:ghost?0x349aff:0xb39168,transparent:true,opacity:ghost?.55:.72});
  function part(w,d,h,x,y,z){const o=new THREE.Mesh(new THREE.BoxGeometry(w,d,h),mat.clone());o.position.set(x,y,z);g.add(o);}
  const w=m.width/100,d=m.depth/100,h=m.height/100,t=.025;
  if(!m.source_object_id){
  for(const x of [-w/2,w/2])part(t,d,h,x,0,h/2);
  for(let i=0;i<=m.levels;i++)part(w,d,t,0,0,h*i/m.levels);
  part(w,t,h,0,d/2,h/2);
  }
  const edge=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w,d,h)),new THREE.LineBasicMaterial({color:0x40aaff,depthTest:false}));edge.position.z=h/2;edge.renderOrder=5;g.add(edge);return g;
 }
 function draw(){for(const o of [...group.children])dispose(o);if(navigation)navigation.colliders=navigation.colliders.filter(b=>!b.manualShelf);for(const s of shelves()){group.add(meshShelf(s));if(navigation&&!s.metadata.source_object_id){const m=s.metadata,a=m.angle*Math.PI/180,c=Math.cos(a),n=Math.sin(a);navigation.colliders.push({manualShelf:true,center:[m.room_position[0],m.room_position[1],data.floor+m.height/200],scale:[m.width/100,m.depth/100,m.height/100],rotation:[[c,-n,0],[n,c,0],[0,0,1]],bottom:data.floor,top:data.floor+m.height/100});}}}
 async function refresh(){entities=await request('/scenes/'+encodeURIComponent(sceneId)+'/entities');data.current_entities=entities;draw();}
 function thumbnail(item){const src=item.metadata?.shelf_photo;if(typeof src==='string'&&/^data:image\/(jpeg|png|webp);base64,/.test(src)){const img=el('img');img.src=src;img.alt=item.name;return img;}return el('span','□','shelf-placeholder');}
 function library(){stopWalking();clearSelection();page('収納');if(editObject)dialog.append(button('認識した家具を修正',()=>{dialog.close();editObject();}));dialog.append(button('収納を順番に撮影',()=>photograph(),'shelf-primary'));
 const list=el('div','','shelf-list');dialog.append(list);
 for(const d of detected){const saved=shelves().find(s=>String(s.metadata.source_object_id)===d.id),target=saved||detectedStorage(d);const b=button(saved? saved.name+' · '+items(saved.id).length+'個'+(saved.metadata.storage_photo?' · 写真あり':''):d.name,()=>useDetected(d));bindPreview(b,target);list.append(b);}
 for(const s of shelves().filter(s=>!s.metadata.source_object_id)){const b=button(s.name+' · '+items(s.id).length+'個',()=>openShelf(s));bindPreview(b,s);list.append(b);}
 if(!list.children.length)list.append(el('p','収納はまだありません。'));
 dialog.append(button('＋ 手動で棚を追加',()=>shelfForm()));
 const first=detected[0]?detectedStorage(detected[0]):shelves()[0];if(first){locateShelf(first);list.firstElementChild?.setAttribute('aria-current','true');}
 }
 function bindPreview(button,target){const show=()=>{locateShelf(target);for(const b of dialog.querySelectorAll('.shelf-list button'))b.removeAttribute('aria-current');button.setAttribute('aria-current','true');};button.addEventListener('pointerenter',e=>{if(e.pointerType==='mouse')show();});button.addEventListener('focus',show);}
 function detectedStorage(d){
 const existing=shelves().find(s=>String(s.metadata.source_object_id)===d.id);if(existing)return existing;
  const collider=data.navigation?.colliders?.find(b=>String(b.id)===d.id);
  const box=new THREE.Box3();for(const mesh of d.item.meshes)box.expandByObject(mesh);
  const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
  const c=collider?.center||center.toArray(),scale=collider?.scale||size.toArray(),rotation=collider?.rotation;
  const angle=rotation?Math.atan2(rotation[1][0],rotation[0][0])*180/Math.PI:0;
  const bottom=collider?.bottom??box.min.z;
  const m={shelf_ui:1,source_object_id:d.id,source_prediction:data.source_prediction,type:labels[d.item.label],room_position:[c[0],c[1],bottom],width:scale[0]*100,depth:scale[1]*100,height:scale[2]*100,angle,levels:4};

 const world=new THREE.Vector3(...m.room_position).applyMatrix4(matrix);
 return {id:'storage-'+sceneId+'-'+encodeURIComponent(data.source_prediction)+'-'+d.id,scene_id:sceneId,name:d.name,kind:'container',source:'human',position:{x:world.x,y:world.y,z:world.z},metadata:m};
 }
 const storagePhotos=installStoragePhotos({targets:()=>[...detected.filter(d=>d.item.label!=='desk').map(detectedStorage),...shelves().filter(s=>!s.metadata.source_object_id)],focus:locateShelf,compress:photoData,done:library,save:async(target,image)=>{const latest=entities.find(e=>e.id===target.id)||target;await save({...latest,metadata:{...latest.metadata,storage_photo:{image,captured_at:new Date().toISOString()}}});await refresh();}});
 document.addEventListener('storage-photograph',()=>photograph());
 function photograph(one){dialog.close();storagePhotos.open(one);}
 function useDetected(d){
  const existing=shelves().find(s=>String(s.metadata.source_object_id)===d.id);if(existing){openShelf(existing);return;}
  const collider=data.navigation?.colliders?.find(b=>String(b.id)===d.id);
  const box=new THREE.Box3();for(const mesh of d.item.meshes)box.expandByObject(mesh);
  const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
  const c=collider?.center||center.toArray(),scale=collider?.scale||size.toArray(),rotation=collider?.rotation;
  const angle=rotation?Math.atan2(rotation[1][0],rotation[0][0])*180/Math.PI:0;
  const bottom=collider?.bottom??box.min.z;
  const m={shelf_ui:1,source_object_id:d.id,source_prediction:data.source_prediction,type:labels[d.item.label],room_position:[c[0],c[1],bottom],width:scale[0]*100,depth:scale[1]*100,height:scale[2]*100,angle,levels:4};
  locateShelf({name:d.name,metadata:m});page(d.name+'を収納として使う');dialog.append(el('p','認識済みの位置・大きさをそのまま使います。中の段数を設定してください。'));
  if(editObject)dialog.append(button('名前変更・誤認識の削除',()=>{dialog.close();editObject(d.id);}));
  const form=el('form');dialog.append(form);const name=field(form,'収納の名前',d.name),levels=field(form,'段数',4,'number',1,12);const submit=el('button','この家具を使う','shelf-primary');submit.type='submit';form.append(submit);
  form.onsubmit=async e=>{e.preventDefault();if(!form.reportValidity())return;submit.disabled=true;try{m.levels=+levels.value;const world=new THREE.Vector3(...m.room_position).applyMatrix4(matrix);const entity={id:'storage-'+sceneId+'-'+encodeURIComponent(data.source_prediction)+'-'+d.id,scene_id:sceneId,name:name.value.trim(),kind:'container',source:'human',position:{x:world.x,y:world.y,z:world.z},metadata:m};if(!entity.name)throw Error('名前を入力してください。');const saved=await save(entity);await refresh();openShelf(saved);}catch(err){message(err.message);}finally{submit.disabled=false;}};
 }
 function shelfForm(){clearSelection();page('棚を追加');const form=el('form');dialog.append(form);
  const name=field(form,'棚の名前','本棚');const lab=el('label','棚の種類'),type=el('select');for(const t of ['本棚','スチール棚','カラーボックス','棚（自由）']){const o=el('option',t);type.append(o);}lab.append(type);form.append(lab);
  const dims=el('div','','shelf-dims');form.append(dims);const width=field(dims,'幅 cm',80,'number',20,500),height=field(dims,'高さ cm',180,'number',20,400),depth=field(dims,'奥行 cm',30,'number',10,200);
  const levels=field(form,'段数',4,'number',1,12),angle=field(form,'向き（度）',0,'number',0,359);
  const submit=el('button','部屋に置く','shelf-primary');submit.type='submit';form.append(submit);
  form.onsubmit=e=>{e.preventDefault();if(!form.reportValidity())return;placing={id:crypto.randomUUID(),scene_id:sceneId,name:name.value.trim(),kind:'container',source:'human',metadata:{shelf_ui:1,type:type.value,width:+width.value,height:+height.value,depth:+depth.value,levels:+levels.value,angle:+angle.value,room_position:null}};if(!placing.name)return;dialog.close();overview();banner.hidden=false;banner.replaceChildren(el('span','床をタップして棚の位置を決めてください'),button('キャンセル',cancelPlacement));};
 }
 function cancelPlacement(){placing=null;if(preview)dispose(preview);preview=null;banner.hidden=true;}
 function locateShelf(s){
  stopWalking();overview();const m=s.metadata;
  const detectedItem=detected.find(d=>String(d.id)===String(m.source_object_id));
  const manual=group.children.find(g=>g.userData.entity?.id===s.id);
  const bounds=storageHighlight.show(detectedItem?.item.meshes||(manual?[manual]:[]),m);
  if(bounds.isEmpty())return;
  highlightedName=s.name||detectedItem?.name||'選択した収納';selectionStatus();
  const target=bounds.getCenter(new THREE.Vector3()),size=bounds.getSize(new THREE.Vector3());
  const footprint=data.footprint||data.navigation?.footprint||[];
  const center=footprint.length?new THREE.Vector3(footprint.reduce((sum,p)=>sum+p[0],0)/footprint.length,footprint.reduce((sum,p)=>sum+p[1],0)/footprint.length,target.z):target.clone();
  const direction=center.sub(target);direction.z=0;if(direction.length()<.1)direction.set(0,-1,0).applyAxisAngle(new THREE.Vector3(0,0,1),(m.angle||0)*Math.PI/180);direction.normalize();
  const mobile=renderer.domElement.clientWidth<=700,distance=Math.max(2.8,size.length()*(mobile?2.05:1.7));
  const aim=target.clone();
  if(mobile)aim.z-=Math.max(.45,size.z*.48);
  else aim.addScaledVector(new THREE.Vector3(-direction.y,direction.x,0),distance*.24);
  camera.position.copy(aim).addScaledVector(direction,distance);camera.position.z+=distance*.32;orbit.target.copy(aim);orbit.update();
 }
 function openShelf(s){selected=s;locateShelf(s);page(s.name);if(editObject&&s.metadata.source_object_id)dialog.append(button('名前変更・誤認識の削除',()=>{dialog.close();editObject(String(s.metadata.source_object_id));}));if(s.metadata.storage_photo?.image){const img=el('img');img.src=s.metadata.storage_photo.image;img.alt=s.name+'の写真';img.className='shelf-detail-photo';dialog.append(img);}dialog.append(button(s.metadata.storage_photo?'収納の写真を撮り直す':'この収納を撮影',()=>photograph(s)));
 dialog.append(button('＋ 物を登録',()=>itemForm(s),'shelf-primary'));
 const list=el('div','','shelf-items');dialog.append(list);
 for(const item of items(s.id)){const b=button('',()=>details(s,item)),label=el('span',item.name);label.append(el('small','上から'+item.metadata.level+'段目 · '+item.metadata.side));b.append(thumbnail(item),label);list.append(b);}
 if(!list.children.length)list.append(el('p','まだ物が登録されていません。'));
 const more=el('details'),summary=el('summary','収納の情報');more.append(summary,el('p',s.metadata.type+' · '+s.metadata.levels+'段'),button(s.metadata.source_object_id?'収納の登録を解除':'棚を削除',async()=>{if(!confirm(s.metadata.source_object_id?'この収納の登録と中の物の記録を削除しますか？認識済みの家具は残ります。':'この棚と、中に登録した物を削除しますか？'))return;try{for(const item of items(s.id))await request('/entities/'+encodeURIComponent(item.id),{method:'DELETE'});await request('/entities/'+encodeURIComponent(s.id),{method:'DELETE'});await refresh();library();}catch(err){message(err.message);}}));dialog.append(more);
 }
 async function photoData(file){if(file.size>15000000)throw Error('写真は15MB以下を選んでください。');const bitmap=await createImageBitmap(file);try{const c=document.createElement('canvas'),ratio=Math.min(1,640/Math.max(bitmap.width,bitmap.height));c.width=Math.round(bitmap.width*ratio);c.height=Math.round(bitmap.height*ratio);c.getContext('2d').drawImage(bitmap,0,0,c.width,c.height);return c.toDataURL('image/jpeg',.8);}finally{bitmap.close();}}
 function itemForm(s,item=null,row=1,side='中央'){locateShelf(s);page(item?'物を編集':'物を登録');const form=el('form');dialog.append(form);const name=field(form,'物の名前',item?.name||'');const level=field(form,'上から何段目',item?.metadata.level||row,'number',1,s.metadata.levels);const label=el('label','左右の位置'),select=el('select');for(const x of ['左','中央','右'])select.append(el('option',x));select.value=item?.metadata.side||side;label.append(select);form.append(label);
  const photoLabel=el('label','写真（任意）'),photo=el('input');photo.type='file';photo.accept='image/*';photoLabel.append(photo);form.append(photoLabel);const note=field(form,'メモ（任意）',item?.metadata.note||'');note.required=false;const submit=el('button','保存','shelf-primary');submit.type='submit';form.append(submit);
  form.onsubmit=async e=>{e.preventDefault();if(!form.reportValidity())return;submit.disabled=true;try{const m=s.metadata;const local=new THREE.Vector3(select.value==='左'?-m.width/300:select.value==='右'?m.width/300:0,-m.depth/200,m.height/100*(1-(+level.value-.5)/m.levels));local.applyAxisAngle(new THREE.Vector3(0,0,1),m.angle*Math.PI/180).add(new THREE.Vector3(...m.room_position));const world=local.applyMatrix4(matrix);const record={...item,id:item?.id||crypto.randomUUID(),scene_id:sceneId,name:name.value.trim(),kind:'object',source:'human',position:{x:world.x,y:world.y,z:world.z},parent_id:s.id,relation_to_parent:'inside',description:s.name+'・上から'+level.value+'段目・'+select.value,tags:['棚の中'],metadata:{shelf_item:1,level:+level.value,side:select.value,note:note.value,shelf_photo:photo.files[0]?await photoData(photo.files[0]):item?.metadata.shelf_photo||null}};if(!record.name)throw Error('名前を入力してください。');await save(record);await refresh();openShelf(s);}catch(err){message(err.message);}finally{submit.disabled=false;}};
 }
 function details(s,item){locateShelf(s);page(item.name);const img=thumbnail(item);img.classList.add('shelf-detail-photo');dialog.append(img,el('p',item.description),el('p',item.metadata.note||''),button('ここへ案内',()=>{dialog.close();finder.showTarget(item);finder.beginGuide();},'shelf-primary'),button('見つけました',()=>{page('見つかりました！');dialog.append(thumbnail(item),el('h3',item.name),el('p',item.description),button('棚に戻る',()=>openShelf(s),'shelf-primary'));}),button('写真・メモを編集',()=>itemForm(s,item)),button('棚に戻る',()=>openShelf(s)));}
 let down=null;const selectionBlocked=()=>document.body.dataset.captureGuide==='true'||Boolean(document.querySelector('dialog[open]'));renderer.domElement.addEventListener('pointerdown',e=>{down=selectionBlocked()?null:[e.clientX,e.clientY];});renderer.domElement.addEventListener('pointercancel',()=>{down=null;});renderer.domElement.addEventListener('pointerup',e=>{const start=down;down=null;if(selectionBlocked()||!start||Math.hypot(e.clientX-start[0],e.clientY-start[1])>8)return;const rect=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(pointer,camera);
  if(placing){const hit=ray.intersectObjects(floorGroup.children,true)[0];if(!hit)return;placing.metadata.room_position=[hit.point.x,hit.point.y,data.floor];if(preview)dispose(preview);preview=meshShelf(placing,true);scene.add(preview);banner.replaceChildren(el('span','この位置に棚を置きますか？'),button('ここに保存',async()=>{const b=banner.querySelectorAll('button')[0];b.disabled=true;try{const world=new THREE.Vector3(...placing.metadata.room_position).applyMatrix4(matrix);placing.position={x:world.x,y:world.y,z:world.z};const s=await save(placing);cancelPlacement();await refresh();openShelf(s);}catch(err){banner.prepend(el('p',err.message));b.disabled=false;}}),button('キャンセル',cancelPlacement));
  }else{const hit=ray.intersectObjects(group.children,true)[0];if(hit){let o=hit.object;while(o&&!o.userData.entity)o=o.parent;if(o)openShelf(o.userData.entity);}else{const hits=ray.intersectObjects(detected.flatMap(d=>d.item.meshes),true);if(hits.length){let mesh=hits[0].object;const d=detected.find(d=>d.item.meshes.some(m=>{for(let o=mesh;o;o=o.parent)if(o===m)return true;return false;}));if(d)useDetected(d);}}}
 });
 document.addEventListener('shelf-open-item',e=>{const item=entities.find(x=>x.id===e.detail);const s=item&&shelves().find(x=>x.id===item.parent_id);if(s)details(s,item);});
 try{await refresh();}catch(err){launcher.textContent='棚を再読み込み';launcher.onclick=async()=>{try{await refresh();launcher.textContent='収納';launcher.onclick=library;library();}catch(e){page('棚');message(e.message);}};}
}
