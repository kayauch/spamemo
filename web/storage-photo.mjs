export function installStoragePhotos({targets,save,focus,done,compress}){
 const modal=document.createElement('dialog');modal.className='shelf-dialog storage-photo-dialog';modal.setAttribute('aria-label','収納を順番に撮影');document.body.append(modal);
 let queue=[],index=0,pending=null,busy=false,version=0;
 const el=(tag,text)=>{const n=document.createElement(tag);if(text)n.textContent=text;return n;};
 function button(text,fn,primary=false){const b=el('button',text);b.type='button';if(primary)b.className='shelf-primary';b.onclick=fn;return b;}
 function close(){if(busy)return;version++;pending=null;modal.close();done();}
 modal.addEventListener('cancel',e=>{e.preventDefault();close();});
 function render(){
  modal.replaceChildren();const head=el('header');head.append(el('h2',index<queue.length?'収納の写真':'撮影終了'),button('閉じる',close));modal.append(head);
  if(index>=queue.length){modal.append(el('p','保存した写真は、それぞれの収納から確認できます。'),button('収納へ戻る',close,true));return;}
  const target=queue[index];focus(target);modal.append(el('p',`${index+1} / ${queue.length}`),el('h3',target.name),el('p','収納を正面から、全体が入るように撮ってください。'));
  const photo=pending||target.metadata?.storage_photo?.image;if(photo){const img=el('img');img.src=photo;img.alt=target.name+'の写真';img.className='shelf-detail-photo';modal.append(img);}
  const error=el('p');error.setAttribute('role','status');modal.append(error);
  async function choose(file){if(!file)return;const token=version;for(const b of modal.querySelectorAll('button'))b.disabled=true;try{const result=await compress(file);if(token!==version)return;pending=result;render();}catch(e){error.textContent=e.message;for(const b of modal.querySelectorAll('button'))b.disabled=false;}}
  const camera=el('input');camera.type='file';camera.accept='image/*';camera.setAttribute('capture','environment');camera.hidden=true;camera.onchange=()=>choose(camera.files[0]);
  const album=el('input');album.type='file';album.accept='image/*';album.hidden=true;album.onchange=()=>choose(album.files[0]);modal.append(camera,album);
  modal.append(button(pending?'撮り直す':'写真を撮る',()=>camera.click(),!pending),button('写真から選ぶ',()=>album.click()));
  if(pending)modal.append(button('保存して次へ',async()=>{if(busy)return;busy=true;for(const b of modal.querySelectorAll('button'))b.disabled=true;error.textContent='保存しています…';try{await save(target,pending);pending=null;index++;render();}catch(e){error.textContent=e.message;for(const b of modal.querySelectorAll('button'))b.disabled=false;}finally{busy=false;}},true));
  modal.append(button('今回はスキップ',()=>{pending=null;index++;render();}));
 }
 return {open(one){version++;queue=one?[one]:targets();index=0;pending=null;render();modal.showModal();}};
}
