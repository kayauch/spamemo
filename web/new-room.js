import {installGenerationProgress} from './generation-progress.mjs?v=20260926-poll2';
import {resolveCaptureEntry,isMissingCaptureParent} from './capture-entry.mjs?v=20260926-recovery1';
import {installRoomWizard} from './room-wizard.mjs?v=20260926-simple1';
import {installCaptureGuide} from './capture-guide.mjs?v=3';
import {FURNITURE_LABELS,captureOptions,validateCaptureFiles,validatePanoramaDimensions,furnitureOverride,safeLocalPath,errorMessage,settingsChanged} from './capture-utils.mjs?v=4';
const $=id=>document.getElementById(id),query=new URLSearchParams(location.search);
let currentScene=query.get('scene'),captureJob=query.get('job'),busy=false,uploading=false,timer=null,previewUrl=null,qualityData=null,requestVersion=0;
let objectEditors=[],pollFailures=0;
const wizard=installRoomWizard();
const generation=installGenerationProgress();
const guide=installCaptureGuide($('guidedCapture'),{isBusy:()=>busy,onChange:()=>{}});
function setBusy(value){busy=value;wizard.setBusy(value);if(value)guide.stop();$('captureFields').disabled=value;$('reviewFields').disabled=value;$('addCapture').disabled=value;}
function entryNotice(message){$('captureNotice').textContent=message;$('captureNotice').hidden=!message;}
function status(message,error=false){$('status').textContent=message;$('status').classList.toggle('capture-error',error);if(error)generation.error(message);else if(message.includes('送信しています')||message.includes('撮影データを確認'))generation.submitting(message);}
function link(id,url){const path=safeLocalPath(url,location.origin);$(id).hidden=!path;if(path)$(id).href=path;else $(id).removeAttribute('href');}
function setLocation(){const q=new URLSearchParams();if(currentScene)q.set('scene',currentScene);if(captureJob)q.set('job',captureJob);history.replaceState(null,'',location.pathname+(q.size?'?'+q:''));}
async function api(url,options={}){const response=await fetch(url,{cache:'no-store',...options});let data;try{data=await response.json();}catch{const error=Error(`サーバーの応答を確認できませんでした (${response.status})。`);error.status=response.status;error.retryable=response.ok||response.status>=500||[408,429].includes(response.status);throw error;}if(!response.ok){const error=Error(errorMessage(data,`処理を確認できませんでした (${response.status})`));error.status=response.status;error.retryable=response.status>=500||[408,429].includes(response.status);throw error;}return data;}

function options(){return captureOptions({room_type:wizard.roomType(),up_axis:$('up').value,ceiling_height_m:$('ceilingHeight').value,reference_length_m:$('referenceLength').value,max_frames:$('maxFrames').value,furniture_frame_budget:$('furnitureFrames').value,texture_quality:$('textureQuality').value,refine_cameras:$('refineCameras').checked});}
function fillOptions(value={}){wizard.fillRoomType(value.room_type);for(const [id,key,fallback] of [['up','up_axis','image-top'],['ceilingHeight','ceiling_height_m',''],['referenceLength','reference_length_m',''],['maxFrames','max_frames',60],['furnitureFrames','furniture_frame_budget',24],['textureQuality','texture_quality','high']])$(id).value=value[key]??fallback;$('refineCameras').checked=Boolean(value.refine_cameras);}
function chooseParent(scene,name){if(!scene)return;if(!Array.from($('parent').options).some(o=>o.value===scene))$('parent').add(new Option(name||'作成した部屋',scene));$('parent').value=scene;if(!$('name').value)$('name').value=$('parent').selectedOptions[0].textContent;}
function resetQuality(){guide.render(null);qualityData=null;objectEditors=[];$('quality').hidden=true;$('reviewStatus').textContent='';}
function make(tag,text,className){const item=document.createElement(tag);if(text!=null)item.textContent=text;if(className)item.className=className;return item;}
function renderQuality(data){
  guide.render(data);qualityData=data;objectEditors=[];$('quality').hidden=false;$('sourceFrames').replaceChildren();$('objects').replaceChildren();$('recommendations').replaceChildren();
  const coverage=data.appearance?.summary?.photo_coverage,parts=[`${data.frame_count||0}枚の撮影画像`,`${data.objects?.length||0}件の家具候補`];
  if(Number.isFinite(coverage))parts.push(`写真を貼れた面積 ${Math.round(coverage*100)}%`);
  parts.push(data.calibration?.metric_scale_calibrated?'入力された実寸で大きさを調整済み':'寸法は未校正');
  const duplicates=data.capture?.duplicates_removed?.length||0;
  if(duplicates)parts.push(`同じ内容の素材${duplicates}件を除外`);
  $('qualitySummary').textContent=parts.join(' · ');
  const messages=new Set((data.recommendations||[]).map(r=>typeof r==='string'?r:r.message).filter(Boolean));
  if(data.camera_refinement){const count=data.camera_refinement.accepted_count||0;messages.add(`カメラ位置の追加調整：${count}枚で採用。条件を満たさない画像は元の位置を使っています。`);}
  for(const object of data.objects||[]){const q=object.evidence?.quality||{};for(const advice of q.needs_photo||[])if(advice.message)messages.add(`${FURNITURE_LABELS[object.label]||object.label}：${advice.message}`);}
  for(const message of messages)$('recommendations').append(make('li',message));
  if(!messages.size)$('recommendations').append(make('li','元の部屋と見比べ、家具の種類・配置と未撮影の面を確認してください。'));
  for(const frame of data.frames||[]){const path=safeLocalPath(frame.url,location.origin);if(!path)continue;const figure=make('figure'),a=make('a'),image=make('img');a.href=path;a.target='_blank';a.rel='noopener';image.src=path;image.loading='lazy';image.alt=`撮影画像 ${frame.index+1}`;a.append(image);figure.append(a,make('figcaption',`画像 ${frame.index+1}${Number.isFinite(frame.timestamp_seconds)?' · '+frame.timestamp_seconds.toFixed(1)+'秒':''}`));$('sourceFrames').append(figure);}
  for(const [i,object] of (data.objects||[]).entries()){
    const card=make('article',null,'capture-object'),label=make('select'),kindLabel=make('label','種類'),hidden=make('input'),hiddenLabel=make('label',null,'capture-check'),dims=make('div',null,'capture-dimensions');
    label.id=`object-kind-${i}`;kindLabel.htmlFor=label.id;for(const [value,text] of Object.entries(FURNITURE_LABELS))label.add(new Option(text,value));
    if(!Object.hasOwn(FURNITURE_LABELS,object.label))label.add(new Option(object.label||'未分類',object.label||''));label.value=object.label;
    hidden.type='checkbox';hidden.checked=Boolean(object.hidden);hiddenLabel.append(hidden,document.createTextNode(object.review_status==='needs_confirmation'?'未確認のため非表示（チェックを外すと採用）':'この家具を非表示にする'));
    const inputs=['幅（m）','奥行（m）','高さ（m）'].map((text,j)=>{const parent=make('label',text),input=make('input');input.type='number';input.min='.03';input.max='10';input.step='.01';input.placeholder='実測値';input.setAttribute('aria-label',`${FURNITURE_LABELS[object.label]||object.label} ${text}`);parent.append(input);dims.append(parent);return input;});
    card.append(make('h3',`${FURNITURE_LABELS[object.label]||object.label} · ${object.id}`),kindLabel,label,hiddenLabel,dims);
    if(object.review_status==='needs_confirmation')card.append(make('p','要確認の家具候補：1か所からの検出です。元画像で確認し、採用すると簡易形状が追加されます。'));
    for(const index of object.source_frames||[]){const frame=(data.frames||[]).find(f=>f.index===index);if(!frame)continue;const path=safeLocalPath(frame.url,location.origin);if(!path)continue;const a=make('a'),img=make('img');a.href=path;a.target='_blank';a.rel='noopener';img.src=path;img.loading='lazy';img.alt=`候補の元画像 ${index+1}`;img.style.width='160px';a.append(img);card.append(a);}
    if(Array.isArray(object.dimensions_m))card.append(make('p','現在の推定寸法：'+object.dimensions_m.map(x=>Number(x).toFixed(2)).join(' × ')+' m（実測確認前）'));
    else card.append(make('p','現在の形状は実寸を確認できていません。'));
    const q=object.evidence?.quality;if(q)card.append(make('p',`${q.frame_count||0}枚 / ${q.independent_viewpoints||0}か所から確認。${q.conflicts?.length?'推定に不一致があります。':''}未撮影の裏側は推定です。`));
    if(object.evidence?.user_corrected)card.append(make('p','利用者による修正を反映しています。'));
    $('objects').append(card);objectEditors.push({object,label,hidden,inputs});
  }
  if(!objectEditors.length)$('objects').append(make('p','確認できる家具候補がありません。家具がある場合は全体と側面を追加撮影してください。','field-hint'));
  $('applyReview').disabled=!data.result_url||['running','queued'].includes(data.state);
}
async function loadQuality(scene=currentScene){if(!scene)return;const version=++requestVersion;try{const data=await api(`/scenes/${encodeURIComponent(scene)}/capture-quality`);if(version!==requestVersion||scene!==currentScene)return;renderQuality(data);fillOptions(data.options);if(data.result_url){link('result',data.result_url);if(data.state==='complete')generation.render({state:'complete',scene_id:scene,result_url:data.result_url});}}catch(error){if(version===requestVersion)$('reviewStatus').textContent=error.message;}}
async function renderJob(data,isCapture){
  generation.render({...data,scene_id:data.scene_id||currentScene});
  if(data.state!=='idle')wizard.setStep(3);
  status(data.message||'処理を確認しています。');if(data.options)fillOptions(data.options);
  if(isCapture&&data.scene_id){currentScene=data.scene_id;chooseParent(currentScene,data.name);setLocation();}
  link('result',data.result_url||data.previous_result_url);link('review',data.review_url);
  if(data.state==='running'||data.state==='queued'){setBusy(true);$('resume').hidden=true;clearTimeout(timer);timer=setTimeout(()=>poll(isCapture),2500);return;}
  setBusy(false);$('resume').hidden=data.state==='complete';
  if(currentScene)await loadQuality();
}
async function poll(isCapture=Boolean(captureJob)){
  clearTimeout(timer);const expected=isCapture?captureJob:currentScene;try{const endpoint=isCapture?`/capture-jobs/${encodeURIComponent(expected)}`:`/scenes/${encodeURIComponent(expected)}/solid-room-job`;const data=await api(endpoint);if(expected!==(isCapture?captureJob:currentScene))return;pollFailures=0;await renderJob(data,isCapture);}catch(error){if(expected!==(isCapture?captureJob:currentScene))return;if(error.retryable||error.status==null){setBusy(true);wizard.setStep(3);$('resume').hidden=true;const message='進捗の通信を再確認しています。生成の停止は確認されていません。撮影データの再送信は不要です。';status(message);generation.reconnecting(message);pollFailures++;timer=setTimeout(()=>poll(isCapture),Math.min(15000,2500*2**Math.min(pollFailures-1,3)));return;}setBusy(false);if(error.status===404){wizard.setStep(3);status('以前の部屋または処理が見つかりません。新しく撮影する場合は「撮影内容に戻る」から進んでください。',true);$('resume').hidden=true;}else{status(`${error.message} 保存した処理は、下のボタンから再確認できます。`,true);$('resume').hidden=false;}}
}
$('resume').onclick=async()=>{if(busy)return;setBusy(true);$('resume').hidden=true;try{if(captureJob)await renderJob(await api(`/capture-jobs/${encodeURIComponent(captureJob)}/resume`,{method:'POST'}),true);else if(currentScene)await renderJob(await api(`/scenes/${encodeURIComponent(currentScene)}/solid-room-job?up_axis=${encodeURIComponent($('up').value)}`,{method:'POST'}),false);}catch(error){setBusy(false);status(error.message,true);$('resume').hidden=false;}};
for(const [id,description] of [['videos','videoFiles'],['photos','photoFiles'],['panoramas','panoramaFiles']])$(id).onchange=()=>{$(description).textContent=Array.from($(id).files).map(f=>f.name).join(' / ');if(id==='videos'){if(previewUrl)URL.revokeObjectURL(previewUrl);const file=$('videos').files[0];$('preview').hidden=!file;if(file){previewUrl=URL.createObjectURL(file);$('preview').src=previewUrl;}else{$('preview').removeAttribute('src');previewUrl=null;}}};
$('clearDepth').onclick=()=>{$('depthBundle').value='';};
$('parent').onchange=async()=>{if(busy)return;clearTimeout(timer);currentScene=$('parent').value||null;captureJob=null;requestVersion++;resetQuality();link('result',null);link('review',null);$('resume').hidden=true;setLocation();if(currentScene){if(!$('name').value)$('name').value=$('parent').selectedOptions[0].textContent;await poll(false);}else status('動画や写真を選んでください。');};
$('addCapture').onclick=()=>{wizard.setStep(2,{focus:true});chooseParent(currentScene);$('form').scrollIntoView({behavior:'smooth',block:'start'});$('videos').focus();status('足りない場所を撮影し、上のフォームから追加してください。以前の結果を残して作成します。');};
async function checkPanorama(file){let bitmap;try{bitmap=await createImageBitmap(file);validatePanoramaDimensions(bitmap.width,bitmap.height);}catch(error){throw Error(`${file.name}：${error.message}`);}finally{bitmap?.close();}}
$('form').onsubmit=async event=>{
  event.preventDefault();if(busy||!wizard.validName())return;if(wizard.currentStep()===1){wizard.setStep(2,{focus:true});return;}
  const files={videos:[...Array.from($('videos').files),...wizard.files()],photos:[...Array.from($('photos').files),...guide.files($('parent').value)],panoramas:Array.from($('panoramas').files),depth_bundle:Array.from($('depthBundle').files),parent_scene_id:$('parent').value};
  try{validateCaptureFiles(files);const settings=options();wizard.setStep(3,{focus:true});setBusy(true);$('resume').hidden=true;status('撮影データを確認しています…');for(const file of files.panoramas)await checkPanorama(file);
    const body=new FormData();for(const key of ['videos','photos','panoramas','depth_bundle'])for(const file of files[key])body.append(key,file);body.append('name',$('name').value.trim()||'新しい部屋');if(files.parent_scene_id)body.append('parent_scene_id',files.parent_scene_id);body.append('options',JSON.stringify(settings));
    status('撮影データを送信しています。送信が終わるまでこの画面を開いていてください。');uploading=true;const data=await api('/capture-jobs',{method:'POST',body});uploading=false;guide.clear(files.parent_scene_id);wizard.clear();captureJob=data.id;currentScene=data.scene_id||null;setLocation();resetQuality();for(const id of ['videos','photos','panoramas','depthBundle'])$(id).value='';for(const id of ['videoFiles','photoFiles','panoramaFiles'])$(id).textContent='';if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null;$('preview').hidden=true;$('preview').removeAttribute('src');await renderJob(data,true);
  }catch(error){setBusy(false);wizard.setStep(3);if(isMissingCaptureParent(error)){status('追加先の部屋は削除されています。動画はこの画面に残っています。「撮影内容に戻る」→「部屋の情報へ」で、追加先を「新しい部屋を作る」に変更して再送してください。',true);}else status(error.message+(uploading&&(!error.status||error.status>=500)?' 送信後に接続が切れた場合、再送の前に部屋一覧で状況を確認してください。':''),true);}finally{uploading=false;}
};
$('reviewForm').onsubmit=async event=>{
  event.preventDefault();if(busy||!currentScene)return;
  try{const settings=options();settings.furniture_overrides=objectEditors.filter(e=>e.label.value!==e.object.label||e.hidden.checked!==Boolean(e.object.hidden)||e.inputs.some(input=>input.value.trim()!=='')).map(e=>furnitureOverride({id:e.object.id,label:e.label.value,hidden:e.hidden.checked,dimensions:e.inputs.map(input=>input.value)}));
    if(!settingsChanged(settings,qualityData?.options||{})){$('reviewStatus').textContent='家具や実寸に変更があるときに使用してください。';return;}
    setBusy(true);$('reviewStatus').textContent='確認内容を保存し、新しい結果を作っています。';const data=await api(`/scenes/${encodeURIComponent(currentScene)}/capture-review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(settings)});captureJob=null;setLocation();await renderJob(data,false);
  }catch(error){setBusy(false);$('reviewStatus').textContent=error.message;}
};
window.addEventListener('beforeunload',event=>{if(uploading||guide.hasPhotos()||wizard.hasRecordings()){event.preventDefault();event.returnValue='';}});
window.addEventListener('pagehide',()=>{if(previewUrl)URL.revokeObjectURL(previewUrl);});
async function initialize(){try{const scenes=await api('/scenes'),entry=resolveCaptureEntry(scenes,currentScene);wizard.setScenes(scenes);for(const scene of scenes)if(!Array.from($('parent').options).some(o=>o.value===scene.id))$('parent').add(new Option(scene.name||'名称未設定',scene.id));if(entry.missing){currentScene=null;$('parent').value='';setLocation();entryNotice('以前の部屋が見つからないため、新しい部屋として作成します。');}else if(currentScene)chooseParent(currentScene);}catch(error){entryNotice(error.message);status(error.message,true);}if(captureJob)await poll(true);else if(currentScene)await poll(false);}
initialize();
