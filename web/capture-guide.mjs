import {FURNITURE_LABELS,safeLocalPath} from './capture-utils.mjs';
export function capturePlan(data){
 const tasks=[],repairs=[],furnitureTasks=[];
 const blocked=new Set((data.recommendations||[]).filter(r=>['projection_depth_mismatch','furniture_mask_occlusion'].includes(r.code)).map(r=>r.surface_id).filter(Boolean));
 for(const r of data.recommendations||[]){
  if(['projection_depth_mismatch','furniture_mask_occlusion','measurement_conflict'].includes(r.code)){if(!repairs.includes(r.message))repairs.push(r.message);continue;}
  if(!['floor','wall','ceiling'].includes(r.kind))continue;
  const key=r.surface_id||r.kind;if(blocked.has(key))continue;if(tasks.some(t=>t.id===key))continue;
  const label={floor:'床',wall:'壁',ceiling:'天井'}[r.kind];
  const instruction=r.kind==='floor'?'床と壁の境目が入り、元の動画と同じ家具や角が見える位置から撮ってください。':r.kind==='ceiling'?'天井と壁の境目を、部屋の角も入れて撮ってください。': '壁全体と隣の壁との角を入れて、正面に近い位置から撮ってください。';
  tasks.push({id:key,title:label+'を追加撮影',instruction,frames:[],note:'この面の正確な撮影位置は自動判定していません。元動画と重なる場所を含めてください。'});
 }
 const furniture=(data.objects||[]).filter(o=>o.review_status==='needs_confirmation'||o.evidence?.quality?.needs_photo?.length);
 for(const o of furniture){
  const frames=o.source_frames||o.evidence?.quality?.source_frames||[];
  furnitureTasks.push({priority:o.review_status==='needs_confirmation',id:'object-'+o.id,title:(FURNITURE_LABELS[o.label]||o.label)+' · '+o.id,instruction:'元画像の家具全体が入る位置まで下がって1枚。その後、横へ移動して斜め・側面からもう1枚撮ってください。',frames,note:'周囲の壁や床も入れてください。対象の種類・範囲は推定なので、元画像と見比べてください。'});
 }
 return {tasks:[...furnitureTasks.filter(t=>t.priority),...tasks,...furnitureTasks.filter(t=>!t.priority)].slice(0,12),repairs};
}
export function captureReceipt(count){return count?`撮影済み ${count}枚・未送信（この画面内に一時保存）`:'まだ撮影していません';}
export function guidedInstruction(task, phase){
 if(!task)return '';
 if(!task.id.startsWith('object-'))return task.instruction;
 return phase===0?'家具全体と周りの床が入る位置まで下がり、止まって撮影してください。':'家具は動かさず、あなたが少し横へ移動し、斜めから撮影してください。';
}
export function installCaptureGuide(root,{onChange,isBusy}){
 let data=null,stream=null,epoch=0,scene=null,active=null,taskList=[],taskIndex=0,phase=0,capturing=false;
 const queue=[];const el=(tag,text)=>{const e=document.createElement(tag);if(text)e.textContent=text;return e;};
 const title=el('h2','写真を追加する'),intro=el('p','カメラ画面の案内に沿って、1枚ずつ撮影します。'),cards=el('div'),repair=el('details'),camera=el('dialog'),prompt=el('p'),video=el('video'),message=el('p'),shots=el('div');
 camera.className='guided-camera';camera.setAttribute('aria-label','ガイド付き撮影');
 const receipt=el('p'),recent=el('img'),receiptPanel=el('div'),queueStatus=el('p');receiptPanel.className='guided-receipt';receipt.setAttribute('role','status');recent.alt='直前に撮った写真';recent.hidden=true;receiptPanel.append(receipt,recent);
 const reference=el('img'),target=el('p'),progress=el('span'),bottom=el('div'),top=el('div'),alternatives=el('details');
 reference.className='guided-reference';reference.alt='撮る対象の元画像';bottom.className='guided-bottom';top.className='guided-top';prompt.className='guided-instruction';alternatives.append(el('summary','撮る対象を選ぶ'),cards);
 cards.className='capture-objects';shots.className='capture-frames';video.autoplay=true;video.muted=true;video.playsInline=true;video.style.maxWidth='100%';camera.hidden=true;message.setAttribute('role','status');
 const button=(text,fn)=>{const b=el('button',text);b.type='button';b.className='secondary-btn';b.onclick=fn;return b;};
 function close(){stop();camera.close();camera.hidden=true;}
 camera.addEventListener('cancel',event=>{event.preventDefault();close();});
 function showTask(index){if(!taskList.length)return;taskIndex=index%taskList.length;active=taskList[taskIndex];phase=0;updatePrompt();}
 function updatePrompt(){target.textContent=active?.title||'';progress.textContent=`対象 ${taskIndex+1} / ${taskList.length} · ${phase+1} / ${active?.id.startsWith('object-')?2:1}枚`;prompt.textContent=guidedInstruction(active,phase);const frame=data?.frames?.find(f=>f.index===active?.frames[0]),url=frame&&safeLocalPath(frame.url,location.origin);reference.hidden=!url;if(url)reference.src=url;else reference.removeAttribute('src');}
 function showCamera(index){if(isBusy())return;showTask(index);camera.hidden=false;if(!camera.open)camera.showModal();message.textContent='カメラを開き、案内に沿って撮影してください。';}
 function stop(){epoch++;stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;video.hidden=true;open.hidden=false;snap.hidden=true;}
 const native=el('input');native.type='file';native.accept='image/jpeg,image/png,image/webp';native.setAttribute('capture','environment');native.hidden=true;
 function add(file){if(!scene||!active)return;if(!file.size){message.textContent='写真を保存できませんでした。もう一度撮影してください。';return;}if(queue.filter(q=>q.scene===scene).length>=32){message.textContent='追加写真は32枚までです。不要な写真を外してください。';return;}queue.push({scene,task:active.id,file,url:URL.createObjectURL(file)});renderShots();onChange();if(active.id.startsWith('object-')&&phase===0){phase=1;updatePrompt();message.textContent='✓ 撮影できました。左上の写真を確認し、次は斜めから撮ってください。';}else if(taskIndex+1<taskList.length){showTask(taskIndex+1);message.textContent='✓ 撮影できました。左上に今の写真を表示しています。次の対象です。';}else{close();message.textContent='撮影が終わりました。写真を確認して送信してください。';shots.scrollIntoView({behavior:'smooth'});}}
 native.onchange=()=>{if(!isBusy()&&native.files[0])add(native.files[0]);native.value='';};
 const open=button('カメラを開く',async()=>{if(isBusy())return;stop();const token=epoch;try{if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia)throw Error('camera unavailable');const next=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1920},height:{ideal:1080}},audio:false});if(token!==epoch){next.getTracks().forEach(t=>t.stop());return;}stream=next;video.srcObject=next;video.hidden=false;await video.play();open.hidden=true;snap.hidden=false;message.textContent='対象全体と周囲を入れて、止まってから撮影してください。';}catch{stop();message.textContent='カメラを開けません。HTTPSで開くか、「スマホのカメラ・写真を使う」を選んでください。';}});
 const snap=button('写真を撮る',()=>{if(capturing)return;if(isBusy()||!stream||!video.videoWidth){message.textContent='先にカメラを開いてください。';return;}capturing=true;snap.disabled=true;const token=epoch,c=el('canvas');c.width=video.videoWidth;c.height=video.videoHeight;c.getContext('2d').drawImage(video,0,0);c.toBlob(blob=>{capturing=false;snap.disabled=false;if(token!==epoch||isBusy())return;if(!blob?.size){message.textContent='撮影に失敗しました。もう一度ボタンを押してください。';return;}add(new File([blob],`guided-${Date.now()}.jpg`,{type:'image/jpeg'}));},'image/jpeg',.94);});
 const improve=button('この写真を使って部屋を改善する',()=>{if(isBusy()||!queue.some(q=>q.scene===scene))return;close();document.getElementById('form').requestSubmit();});
 top.append(progress,button('閉じる',close));
 bottom.append(target,prompt,open,snap,message,button('この対象をスキップ',()=>{if(capturing)return;if(taskIndex+1<taskList.length)showTask(taskIndex+1);else close();}),button('撮った写真を確認・送信',()=>{close();shots.scrollIntoView({behavior:'smooth',block:'center'});}));
 const fallback=el('details');fallback.append(el('summary','カメラが使えないとき'),button('スマホのカメラ・写真を使う',()=>{if(!isBusy())native.click();}));bottom.append(fallback);
 camera.append(video,top,reference,receiptPanel,bottom,native);document.body.append(camera);
 const launch=button('撮影画面を開く',()=>showCamera(0));launch.className='primary-btn';
 root.append(title,intro,launch,repair,alternatives,queueStatus,shots,improve);
 function renderShots(){const pending=queue.filter(q=>q.scene===scene);receipt.textContent=captureReceipt(pending.length);queueStatus.textContent=pending.length?`${pending.length}枚を撮影済みです。まだ送信されていません。下のボタンで送信してください。ページを閉じると写真は失われます。`:'撮った写真はここに表示されます。';recent.hidden=!pending.length;if(pending.length)recent.src=pending.at(-1).url;else recent.removeAttribute('src');shots.replaceChildren();for(const q of queue.filter(q=>q.scene===scene)){const f=el('figure'),img=el('img');img.src=q.url;img.alt='追加する写真';f.append(img,button('この写真を外す',()=>{if(isBusy())return;queue.splice(queue.indexOf(q),1);URL.revokeObjectURL(q.url);renderShots();onChange();}));shots.append(f);}improve.disabled=isBusy()||!queue.some(q=>q.scene===scene);}
 function render(value){close();data=value;scene=data?.scene_id||null;active=null;camera.hidden=true;root.hidden=!scene;cards.replaceChildren();repair.replaceChildren();if(!data){renderShots();return;}const plan=capturePlan(data);taskList=plan.tasks;launch.disabled=!taskList.length;repair.hidden=!plan.repairs.length;if(plan.repairs.length){repair.append(el('summary','形状の確認が必要な箇所'),...plan.repairs.map(t=>el('p',t)));}for(const task of plan.tasks){const card=el('article');card.className='capture-object';card.append(el('h3',task.title),el('p',task.instruction),el('p',task.note));for(const index of task.frames.slice(0,2)){const frame=data.frames?.find(f=>f.index===index),url=frame&&safeLocalPath(frame.url,location.origin);if(url){const img=el('img');img.src=url;img.alt=task.title+'の元画像';img.style.width='100%';img.loading='lazy';card.append(img);}}card.append(button('この対象を撮る',()=>showCamera(taskList.indexOf(task))));cards.append(card);}if(!plan.tasks.length)cards.append(el('p','追加撮影の対象を特定できませんでした。上の確認項目と元写真を比較してください。'));renderShots();}
 document.addEventListener('visibilitychange',()=>{if(document.hidden)close();});window.addEventListener('pagehide',close);
 return {render,stop:close,files:id=>queue.filter(q=>q.scene===id).map(q=>q.file),clear:id=>{for(let i=queue.length-1;i>=0;i--)if(queue[i].scene===id){URL.revokeObjectURL(queue[i].url);queue.splice(i,1);}renderShots();},hasPhotos:()=>queue.length>0};
}
