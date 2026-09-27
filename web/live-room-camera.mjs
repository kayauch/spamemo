import {LiveCaptureSession,guidedRevisionBody} from './live-capture-session.mjs?v=repair-capture2';
import {validateCaptureFiles,errorMessage} from './capture-utils.mjs?v=repair-capture2';
import {MAX_REPAIR_PHOTOS,shotKey,photoOrientation} from './repair-capture-plan.mjs?v=repair-capture2';

export function installLiveRoomCamera({sceneId,run,result,showTarget,previewImage,mediaDevices=navigator.mediaDevices}){
  const modal=document.createElement('dialog');modal.className='live-room-camera';modal.setAttribute('aria-label','部屋の修正・案内に沿って撮影');
  modal.innerHTML=`<section class="live-shoot"><video autoplay muted playsinline aria-label="カメラ映像"></video><header class="live-header"><span class="live-count"></span><button type="button" data-action="review">写真を確認・終了</button></header><section class="live-prompt" aria-live="polite"><div><h2></h2><p class="live-instruction"></p><p class="live-warning" hidden>形状も確認が必要です。写真追加だけでは直らない場合があります。</p></div><button type="button" class="live-preview-button" aria-label="3Dの目印を拡大" aria-expanded="false"><img class="live-target-image" alt="青い部分が次に撮る場所です"><span>目印を拡大</span></button></section><div class="live-camera-message"><p class="live-orientation" role="status" hidden></p><p class="live-message" role="status"></p><button type="button" data-action="open">カメラを開く</button><button type="button" data-action="native">スマホのカメラ・写真を使う</button></div><footer class="live-controls"><button type="button" data-action="skip">今は撮れない</button><button type="button" data-action="snap" aria-label="横向きの写真を撮って次へ" disabled><span></span></button><span class="live-receipt" role="status">0枚撮影</span></footer><p class="live-camera-note">横向きで撮影 · 写真の一致や改善は、再生成後に確認します</p><input type="file" accept="image/jpeg,image/png,image/webp" capture="environment" hidden></section><section class="live-review" hidden><button type="button" data-action="close" class="live-back">← 部屋の修正に戻る</button><h2>撮った写真を確認</h2><p class="live-review-note"></p><p class="live-budget-note"></p><div class="live-photos"></div><button type="button" data-action="continue">未撮影の場所を続ける</button><button type="button" data-action="submit" class="room-primary">この写真で再生成</button><p class="live-send-status" role="status"></p><a class="live-check-jobs" href="/rooms" target="_blank" rel="noopener" hidden>別のタブで部屋一覧を確認</a><p class="live-review-footnote">元の部屋は残ります。写真は送信するまで、このページ内に一時保存されます。撮影済みは写真を受け取った数で、貼り付け成功数ではありません。</p></section>`;
  document.body.append(modal);
  const $=selector=>modal.querySelector(selector),action=name=>$(`[data-action="${name}"]`),video=$('video');
  let session=null,stream=null,epoch=0,capturing=false,sending=false,accepted=false,scene=null,quality=null,load=null;
  const urls=new Map(),landscape=window.matchMedia('(orientation: landscape)');
  const url=file=>{if(!urls.has(file))urls.set(file,URL.createObjectURL(file));return urls.get(file);};
  const revoke=file=>{if(urls.has(file)){URL.revokeObjectURL(urls.get(file));urls.delete(file);}};
  const limit=()=>quality?.photo_budget?.available===false?0:Math.min(MAX_REPAIR_PHOTOS,quality?.photo_budget?.max_additional_photos??MAX_REPAIR_PHOTOS);
  const replacesCurrent=()=>session?.current&&session.shots.has(shotKey(session.current));
  const atLimit=()=>!!session&&session.photos.length>=limit()&&!replacesCurrent();
  function message(text){$('.live-message').textContent=text;}
  function orientation(){return photoOrientation({landscape:landscape.matches,width:video.videoWidth,height:video.videoHeight});}
  function updateButtons(){
    const state=orientation(),active=modal.open&&!$('.live-shoot').hidden;
    $('.live-orientation').hidden=!active||!stream||state.allowed;
    $('.live-orientation').textContent=state.message;
    if(stream){action('open').hidden=state.reason!=='portrait-frame';action('open').textContent=state.reason==='portrait-frame'?'横向きでカメラを開き直す':'カメラを開く';}
    action('snap').disabled=capturing||sending||!stream||video.readyState<2||!session?.current||!state.allowed||atLimit();
    action('skip').disabled=capturing||sending;action('native').disabled=capturing||sending||atLimit();
    action('submit').disabled=sending||!session?.photos.length||!scene||!quality||session.photos.length>limit();
  }
  function stop(){epoch++;stream?.getTracks().forEach(track=>track.stop());stream=null;video.srcObject=null;action('open').hidden=false;action('native').hidden=false;updateButtons();}
  function renderTarget(){
    if(!session?.current){review();return;}
    $('.live-preview-button').setAttribute('aria-expanded','false');$('.live-preview-button span').textContent='目印を拡大';$('.live-preview-button').setAttribute('aria-label','3Dの目印を拡大');
    const target=session.current;$('.live-count').textContent=`${session.photos.length}枚撮影 · 未撮影${session.pending.length}枚 · 上限${limit()}枚`;
    $('.live-prompt h2').textContent=(session.retake?'撮り直し：':'次は、')+target.title;
    $('.live-instruction').textContent=target.instruction;
    $('.live-warning').hidden=!target.repair;$('.live-receipt').textContent=(session.photos.length?'✓ ':'')+session.photos.length+'枚撮影';
    showTarget(target.key,target);try{$('.live-target-image').src=previewImage();$('.live-target-image').hidden=false;}catch{$('.live-target-image').hidden=true;}
    if(atLimit())message(`今回は${limit()}枚まで追加できます。未撮影${session.pending.length}枚の案内は残しています。写真を確認して再生成してください。`);
    updateButtons();
  }
  async function openCamera(){
    stop();const token=epoch;action('open').disabled=true;message('カメラを開いています…');
    try{
      if(!mediaDevices?.getUserMedia)throw Error('unavailable');
      const next=await mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1920},height:{ideal:1080}},audio:false});
      if(token!==epoch){next.getTracks().forEach(t=>t.stop());return;}
      stream=next;video.srcObject=next;await video.play();if(token!==epoch)return;
      for(const track of next.getVideoTracks())track.addEventListener('ended',()=>{if(stream===next){stop();message('カメラが停止しました。もう一度開くと続けられます。');}});
      action('open').hidden=true;action('native').hidden=true;message('');renderTarget();
    }catch{if(token===epoch){stop();message('カメラを開けません。カメラの許可を確認するか、スマホのカメラ・写真を使ってください。');}}
    finally{action('open').disabled=false;}
  }
  function shoot(){
    if(capturing||sending||!session?.current||!stream||video.readyState<2||!orientation().allowed||atLimit())return;
    capturing=true;updateButtons();const token=epoch,canvas=document.createElement('canvas'),scale=Math.min(1,1920/video.videoWidth);
    canvas.width=Math.round(video.videoWidth*scale);canvas.height=Math.round(video.videoHeight*scale);
    try{canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);canvas.toBlob(blob=>{
      capturing=false;if(token!==epoch){updateButtons();return;}
      if(!blob?.size){message('写真を保存できませんでした。もう一度撮影してください。');updateButtons();return;}
      accept(new File([blob],`room-photo-${Date.now()}.jpg`,{type:'image/jpeg'}));
    },'image/jpeg',.92);}catch{capturing=false;message('撮影できませんでした。もう一度お試しください。');updateButtons();}
  }
  function accept(file){
    try{
      const target=session.current;if(!target)return;
      if(atLimit())throw Error(`この部屋には今回は${limit()}枚まで追加できます。撮った写真は残っています。`);
      validateCaptureFiles({photos:[...session.photos.filter(p=>shotKey(p.target)!==shotKey(target)).map(p=>p.file),file],parent_scene_id:sceneId});
      const old=session.commit(file);if(old)revoke(old.file);message('');renderTarget();
    }catch(error){message(error.message);updateButtons();}
  }
  function review(){
    stop();$('.live-shoot').hidden=true;$('.live-review').hidden=false;
    const container=$('.live-photos');container.replaceChildren();
    for(const {file,target} of session?.photos||[]){
      const card=document.createElement('article'),img=document.createElement('img'),title=document.createElement('strong'),retry=document.createElement('button'),remove=document.createElement('button');
      img.src=url(file);img.alt=target.title+'の追加写真';title.textContent=target.title;
      retry.type=remove.type='button';retry.textContent='撮り直す';remove.textContent='外す';retry.disabled=remove.disabled=sending;
      retry.onclick=()=>{if(sending)return;session.retry(shotKey(target));showCamera();};remove.onclick=()=>{if(sending)return;session.shots.delete(shotKey(target));revoke(file);review();};
      card.append(img,title,retry,remove);if(target.kind==='wall'){const direct=document.createElement('button');direct.type='button';direct.textContent='壁への貼り方を調整';direct.disabled=sending;direct.onclick=()=>document.dispatchEvent(new CustomEvent('wall-photo-open',{detail:{key:target.key,file}}));card.append(direct);}container.append(card);
    }
    if(!session?.photos.length)container.textContent='撮った写真はまだありません。';
    $('.live-review h2').textContent=`撮った写真を確認（${session?.photos.length||0}枚）`;
    $('.live-review-note').textContent=`撮影済み ${session?.photos.length||0} / ${session?.targets.length||0}枚。未撮影 ${session?.pending.length||0}枚（スキップした場所を含む）。まだ送信していません。撮れない場所があれば、ここまでの写真で再生成できます。`;
    $('.live-budget-note').textContent=session?.targets.length>limit()?`今回は最大${limit()}枚追加できます。既存の撮影も使うための処理上限です。未撮影の案内を省略せず表示しています。`:`横向きで、壁・床・天井は区画ごと、家具は全体と左右から撮影します。`;
    action('continue').disabled=sending||!session?.pending.length||session.photos.length>=limit();action('close').disabled=sending;updateButtons();
  }
  function showCamera(){if(sending)return;$('.live-review').hidden=true;$('.live-shoot').hidden=false;renderTarget();if(session.current)openCamera();}
  async function loadRoom(){
    try{
      const responses=await Promise.all([fetch('/viewer/state/'+encodeURIComponent(sceneId)),fetch('/scenes/'+encodeURIComponent(sceneId)+'/capture-quality')]);
      if(responses.some(r=>!r.ok))throw Error();const [state,report]=await Promise.all(responses.map(r=>r.json()));
      if(state.scene?.id!==sceneId)throw Error();scene=state.scene;quality=report;$('.live-send-status').textContent='';
    }catch{$('.live-send-status').textContent='部屋の情報を読み込めませんでした。部屋の修正に戻って開き直すと再試行できます。写真は残っています。';load=null;}
    if(modal.open){if($('.live-shoot').hidden)review();else renderTarget();}updateButtons();
  }
  async function submit(){
    if(sending)return;let body;
    try{if(session.photos.length>limit())throw Error(`今回は${limit()}枚まで追加できます。写真を選び直してください。`);body=guidedRevisionBody({scene,quality,session,run,result});}catch(error){$('.live-send-status').textContent=error.message;return;}
    sending=true;review();$('.live-send-status').textContent='写真を送信しています。この画面を開いたままお待ちください。';$('.live-check-jobs').hidden=true;
    try{
      let response;try{response=await fetch('/capture-jobs',{method:'POST',body});}catch{throw Error('受付を確認できませんでした。写真は残っています。再送信する前に別のタブで部屋一覧を確認してください。');}
      const result=await response.json().catch(()=>null);
      if(!response.ok)throw Error(errorMessage(result,'写真を送信できませんでした。少し待ってから再度お試しください。'));
      if(!result?.id)throw Error('受付番号を確認できません。再送信する前に部屋一覧を確認してください。');
      accepted=true;location.assign('/room/new?job='+encodeURIComponent(result.id));
    }catch(error){sending=false;review();$('.live-send-status').textContent=error.message;$('.live-check-jobs').hidden=false;}
  }
  function close(){if(sending)return;stop();modal.close();}
  $('.live-preview-button').onclick=()=>{const expanded=$('.live-preview-button').getAttribute('aria-expanded')!=='true';$('.live-preview-button').setAttribute('aria-expanded',String(expanded));$('.live-preview-button span').textContent=expanded?'目印を閉じる':'目印を拡大';$('.live-preview-button').setAttribute('aria-label',expanded?'3Dの目印を閉じる':'3Dの目印を拡大');};
  action('open').onclick=openCamera;action('snap').onclick=shoot;action('review').onclick=review;action('close').onclick=close;action('submit').onclick=submit;
  action('skip').onclick=()=>{if(capturing)return;session.skip();renderTarget();};
  action('continue').onclick=()=>{session.resume();showCamera();};action('native').onclick=()=>{$('input').click();};
  $('input').onchange=async()=>{
    const file=$('input').files[0];$('input').value='';if(sending||capturing||!session?.current||!file)return;
    const token=epoch;capturing=true;updateButtons();
    try{
      const image=await createImageBitmap(file,{imageOrientation:'from-image'}),width=image.width,height=image.height;image.close();
      if(token!==epoch)return;
      if(width<=height)throw Error('横向きの写真を選んでください。今の写真は追加していません。撮影する場合はスマホを横向きにしてください。');
      accept(file);
    }catch(error){if(token===epoch)message(error.message||'写真を読み込めませんでした。別の写真を選んでください。');}
    finally{capturing=false;updateButtons();}
  };
  for(const type of ['loadeddata','playing','resize'])video.addEventListener(type,updateButtons);
  window.addEventListener('resize',updateButtons);window.addEventListener('orientationchange',updateButtons);landscape.addEventListener('change',updateButtons);
  modal.addEventListener('cancel',event=>{event.preventDefault();if(!sending)review();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&stream&&modal.open)review();});
  window.addEventListener('pagehide',stop);
  window.addEventListener('beforeunload',event=>{if(!accepted&&(session?.photos.length||sending)){event.preventDefault();event.returnValue='';}});
  return {open(targets){if(sending)return;load??=loadRoom();if(!session||!session.photos.length)session=new LiveCaptureSession(targets);else session.addTargets(targets);if(!modal.open)modal.showModal();if(session.photos.length)review();else showCamera();}};
}
