import {decorateIcons} from './spamemo-shell.mjs';
decorateIcons();
const $=id=>document.getElementById(id);
for(const [button,dialog] of [['chooseRoom','roomPicker'],['openSettings','viewerSettings'],['openMap','mapDialog']])$(button).onclick=()=>$(dialog).showModal();
for(const dialog of document.querySelectorAll('dialog')){
  dialog.querySelector('[data-close-dialog]').onclick=()=>dialog.close();
  dialog.addEventListener('click',e=>{if(e.target!==dialog)return;const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();});
}
$('focusSearch').onclick=()=>{if(!$('routeBanner').hidden)$('stopGuide').click();$('roomSearch').hidden=false;$('findQuery').focus();};
for(const id of ['inside','overview'])$(id).addEventListener('click',()=>$('mapDialog').close());
// The room engine enables actions only when the real model and coordinates are ready.
$('findQuery').disabled=true;$('roomSearch').querySelector('[type=submit]').disabled=true;$('openMap').disabled=true;
for(const button of document.querySelectorAll('[data-walk-key]'))button.disabled=true;
try{
  const r=await fetch('/scenes');if(!r.ok)throw Error('部屋一覧を読み込めませんでした');
  const rooms=await r.json();for(const room of rooms){const a=document.createElement('a');a.textContent=room.name||'名称未設定';a.href='/?scene='+encodeURIComponent(room.id)+'#room';a.dataset.sceneId=room.id;$('roomChoices').append(a);
    fetch('/scenes/'+encodeURIComponent(room.id)+'/solid-room-job').then(r=>r.ok?r.json():null).then(job=>{const path=job?.result_url||job?.previous_result_url;if(!path)return;const url=new URL(path,location.origin);if(url.origin===location.origin&&url.pathname==='/static/solid-room.html')a.href=url.pathname+url.search;}).catch(()=>{});
  }
  if(!rooms.length)$('roomChoices').textContent='作成済みの部屋はありません。';
}catch(error){$('roomChoices').textContent=error.message;}
