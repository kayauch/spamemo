import * as THREE from 'three';
import {warpPixels,surfaceSignature} from './wall-photo-math.mjs?v=1';

export async function installWallPhotos({data,run,result,surfaces,stopWalking}){
 const sceneId=data.transform?.scene_id;if(!sceneId)return;
 const walls=surfaces.filter(m=>m.userData.kind==='wall'),overlays=new Map();let entries=[],loaded=false;
 const style=document.createElement('link');style.rel='stylesheet';style.href='/static/wall-photo.css?v=1';document.head.append(style);
 const dialog=document.createElement('dialog');dialog.className='wall-photo';dialog.setAttribute('aria-label','壁に写真を貼る');document.body.append(dialog);
 let mesh=null,photo=null,quad=[],patch=null,busy=false,mode='corners',token=0,erase=false;
 function el(tag,text){const n=document.createElement(tag);if(text)n.textContent=text;return n;}
 function btn(text,fn){const b=el('button',text);b.type='button';b.onclick=fn;return b;}
 const status=el('p');status.setAttribute('role','status');
 const canvas=el('canvas'),preview=el('canvas');canvas.setAttribute('aria-label','写真内の壁の四隅を順にタップ');preview.setAttribute('aria-label','貼り付ける写真。消すモードではなぞって除外');
 const close=btn('閉じる',()=>{if(!busy){token++;dialog.close();}});
 const head=el('header');head.append(el('h2','壁に写真を貼る'),close);
 const title=el('p'),note=el('p','壁全体が写った写真を選び、左上・右上・右下・左下の順にタップ。隠れている角は無理に指定しないでください。');
 const input=el('input');input.type='file';input.accept='image/jpeg,image/png,image/webp';input.hidden=true;
 const camera=btn('撮影',()=>{input.setAttribute('capture','environment');input.click();}),choose=btn('写真を選ぶ',()=>{input.removeAttribute('capture');input.click();});
 const existing=el('select');existing.setAttribute('aria-label','撮影済みの写真');existing.add(new Option('撮影済みから選ぶ',''));
 for(const p of [...(data.appearance_processing?.photo_sources||[])].sort((a,b)=>(b.source_kind==='photo')-(a.source_kind==='photo')))existing.add(new Option((p.source_kind==='photo'?'追加写真 ':'動画の写真 ')+(p.index+1),String(p.index)));
 const restart=btn('四隅を選び直す',()=>{if(busy)return;quad=[];patch=null;mode='corners';render();});
 const flip=btn('左右を反転',()=>{if(!patch||busy)return;const c=el('canvas');c.width=patch.width;c.height=patch.height;const ctx=c.getContext('2d');ctx.translate(c.width,0);ctx.scale(-1,1);ctx.drawImage(patch,0,0);patch=c;render();});
 const eraseButton=btn('家具・窓などを消す',()=>{erase=!erase;eraseButton.setAttribute('aria-pressed',String(erase));status.textContent=erase?'写真の上をなぞると、その部分を貼らずに残せます。':'プレビューを確認して保存してください。';});
 const save=btn('この壁に保存',persist),restore=btn('追加写真を外す',remove);
 const actions=el('div');actions.className='wall-photo-actions';actions.append(restart,flip,eraseButton,save,restore);
 dialog.append(head,title,note,camera,choose,existing,input,canvas,preview,actions,status);
 function message(text){status.textContent=text;}
 function entryFor(m){return entries.find(e=>e.metadata?.wall_photo?.signature===surfaceSignature(m.userData)&&e.metadata.wall_photo.run===run);}
 async function apply(m,url){
  const tex=await new THREE.TextureLoader().loadAsync(url);tex.colorSpace=THREE.SRGBColorSpace;
  const old=overlays.get(m);if(old){m.remove(old);old.material.map.dispose();old.material.dispose();}
  const layer=new THREE.Mesh(m.geometry,new THREE.MeshBasicMaterial({map:tex,transparent:true,side:THREE.DoubleSide,depthWrite:false,polygonOffset:true,polygonOffsetFactor:-2,polygonOffsetUnits:-2}));
  layer.renderOrder=4;m.add(layer);overlays.set(m,layer);
 }
 async function request(url,options){const r=await fetch(url,options);if(!r.ok)throw Error('保存・読み込みに失敗しました。写真は残っています。');return r.json();}
 function render(){
  canvas.hidden=!photo||mode!=='corners';preview.hidden=!patch||mode!=='preview';
  save.disabled=busy||!patch;flip.disabled=eraseButton.disabled=!patch||busy;restart.disabled=busy||!photo;close.disabled=busy;camera.disabled=choose.disabled=existing.disabled=busy;restore.hidden=!mesh||!entryFor(mesh);
  if(photo&&mode==='corners'){canvas.width=photo.width;canvas.height=photo.height;const c=canvas.getContext('2d');c.drawImage(photo,0,0);c.strokeStyle='#008bff';c.lineWidth=4;c.beginPath();quad.forEach(([x,y],i)=>{i?c.lineTo(x*canvas.width,y*canvas.height):c.moveTo(x*canvas.width,y*canvas.height);});c.stroke();quad.forEach(([x,y],i)=>{c.fillStyle='#007aff';c.beginPath();c.arc(x*canvas.width,y*canvas.height,13,0,Math.PI*2);c.fill();c.fillStyle='white';c.font='bold 18px sans-serif';c.textAlign='center';c.fillText(String(i+1),x*canvas.width,y*canvas.height+6);});}
  if(patch){preview.width=patch.width;preview.height=patch.height;preview.getContext('2d').drawImage(patch,0,0);}
 }
 async function loadPhoto(blob){
  const epoch=++token;try{if(blob.size>25*1024*1024)throw Error('写真は25MB以下にしてください。');const bitmap=await createImageBitmap(blob,{imageOrientation:'from-image'});if(epoch!==token){bitmap.close();return;}
   photo=el('canvas');const s=Math.min(1,1600/Math.max(bitmap.width,bitmap.height));photo.width=Math.round(bitmap.width*s);photo.height=Math.round(bitmap.height*s);photo.getContext('2d').drawImage(bitmap,0,0,photo.width,photo.height);bitmap.close();quad=[];patch=null;mode='corners';erase=false;message('1：写真の壁の左上をタップしてください。');render();
  }catch(e){message(e.message||'写真を読み込めませんでした。');}
 }
 input.onchange=()=>{if(input.files[0])loadPhoto(input.files[0]);input.value='';};
 existing.onchange=async()=>{if(existing.value==='')return;const epoch=++token;try{const r=await fetch('/scenes/'+encodeURIComponent(sceneId)+'/frames/'+encodeURIComponent(existing.value));if(!r.ok)throw Error('写真が見つかりません。');const b=await r.blob();if(epoch===token)await loadPhoto(b);}catch(e){message(e.message);}};
 canvas.onpointerdown=event=>{
  if(busy||!photo||mode!=='corners')return;const r=canvas.getBoundingClientRect();quad.push([(event.clientX-r.left)/r.width,(event.clientY-r.top)/r.height]);
  if(quad.length===4){try{const aspect=new THREE.Vector3(...mesh.userData.u).length()/new THREE.Vector3(...mesh.userData.v).length(),w=Math.max(64,Math.round(Math.min(1024,1024*aspect))),h=Math.max(64,Math.min(1024,Math.round(w/aspect)));const source=photo.getContext('2d').getImageData(0,0,photo.width,photo.height);
    const pixels=warpPixels(source.data,photo.width,photo.height,quad,w,h);patch=el('canvas');patch.width=w;patch.height=h;patch.getContext('2d').putImageData(new ImageData(pixels,w,h),0,0);mode='preview';message('壁を正面から見た写真です。向きと、家具の写り込みを確認して保存してください。');
   }catch(e){quad=[];message(e.message);}
  }else message(`${quad.length+1}：${['左上','右上','右下','左下'][quad.length]}をタップしてください。`);render();
 };
 let erasing=false;
 function eraseAt(e){if(!erase||!patch||busy)return;const r=preview.getBoundingClientRect(),c=patch.getContext('2d');c.save();c.globalCompositeOperation='destination-out';c.beginPath();c.arc((e.clientX-r.left)/r.width*patch.width,(e.clientY-r.top)/r.height*patch.height,Math.max(patch.width,patch.height)*.035,0,Math.PI*2);c.fill();c.restore();render();}
 preview.onpointerdown=e=>{erasing=true;preview.setPointerCapture(e.pointerId);eraseAt(e);};preview.onpointermove=e=>{if(erasing)eraseAt(e);};preview.onpointerup=preview.onpointercancel=()=>{erasing=false;};
 async function persist(){
  if(busy||!patch||!mesh)return;if(!loaded){message('保存済みの写真を確認できません。画面を開き直してください。');return;}busy=true;render();message('保存しています…');const target=mesh;
  try{const prior=entryFor(target),position=new THREE.Vector3(...target.userData.origin).applyMatrix4(new THREE.Matrix4().set(...data.transform.generation_world_to_vg_world.flat()));
   const value={...(prior||{}),id:prior?.id||crypto.randomUUID(),scene_id:sceneId,name:'壁の追加写真 '+target.userData.id,kind:'region',source:'human',position:{x:position.x,y:position.y,z:position.z},metadata:{...(prior?.metadata||{}),wall_photo:{version:1,run,result,surface_id:target.userData.id,signature:surfaceSignature(target.userData),image:patch.toDataURL('image/png'),source_image:photo.toDataURL('image/jpeg',.9),quad,method:'user_confirmed_planar_photo',observed_geometry:false}}};
   const saved=await request('/entities',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});entries=entries.filter(e=>e.id!==saved.id);entries.push(saved);await apply(target,saved.metadata.wall_photo.image);message('保存しました。3Dに戻ると、この壁に反映されています。');
  }catch(e){message(e.message);}finally{busy=false;render();}
 }
 async function remove(){const prior=entryFor(mesh);if(!prior||busy)return;busy=true;render();try{await request('/entities/'+encodeURIComponent(prior.id),{method:'DELETE'});entries=entries.filter(e=>e.id!==prior.id);const layer=overlays.get(mesh);if(layer){mesh.remove(layer);layer.material.map.dispose();layer.material.dispose();overlays.delete(mesh);}message('追加写真を外しました。元の壁の表示に戻りました。');}catch(e){message(e.message);}finally{busy=false;render();}}
 function open(key,file){if(busy)return;const selected=walls.find(m=>'surface:'+m.userData.id===key);if(!selected)return;stopWalking();token++;mesh=selected;photo=null;patch=null;quad=[];mode='corners';title.textContent='対象：壁 '+(walls.indexOf(mesh)+1);message('撮影するか、保存済みの写真を選んでください。');existing.value='';render();if(!dialog.open)dialog.showModal();if(file)loadPhoto(file);}
 document.addEventListener('wall-photo-open',event=>open(event.detail?.key,event.detail?.file));
 dialog.addEventListener('cancel',event=>{if(busy)event.preventDefault();else token++;});
 try{entries=await request('/scenes/'+encodeURIComponent(sceneId)+'/entities');loaded=true;for(const m of walls){const e=entryFor(m);if(e&&/^data:image\/png;base64,/.test(e.metadata.wall_photo.image))await apply(m,e.metadata.wall_photo.image);}}catch{console.warn('壁の追加写真を読み込めませんでした');}
 return {open};
}
