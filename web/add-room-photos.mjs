import {captureTargets} from './capture-targets.mjs?v=20260926-target2';
import {photoAdvice,revisionBody} from './photo-revision.mjs?v=20260926-target2';
import {captureOptions,validateCaptureFiles,safeLocalPath,errorMessage} from './capture-utils.mjs?v=4';
const $=id=>document.getElementById(id),sceneId=new URLSearchParams(location.search).get('scene');
let captureTarget=null;
let scene,quality,photos=[],busy=false,ready=false,accepted=false;
function status(message,error=false){$('status').textContent=message;$('status').dataset.error=String(error);}
async function read(url){const r=await fetch(url,{cache:'no-store'}),data=await r.json().catch(()=>null);if(!r.ok||!data)throw Error(errorMessage(data,'部屋の情報を読み込めませんでした。'));return data;}
function render(){
  $('photoControls').disabled=!ready||busy;
  $('submit').disabled=!ready||busy||!photos.length;
  $('submit').textContent=busy?'写真を送信しています…':'写真を追加して再生成';
  $('photoCount').textContent=photos.length+'枚';$('empty').hidden=!!photos.length;
  $('photos').replaceChildren();
  photos.forEach((entry,index)=>{
    const li=document.createElement('li'),img=document.createElement('img'),name=document.createElement('span'),remove=document.createElement('button');
    img.src=entry.url;img.alt='追加写真 '+(index+1);name.className='filename';name.textContent=entry.file.name;
    remove.type='button';remove.textContent='×';remove.setAttribute('aria-label',`写真${index+1}を取り除く`);
    remove.onclick=()=>{URL.revokeObjectURL(entry.url);photos.splice(index,1);render();status(photos.length?photos.length+'枚を追加できます。':'写真を撮るか、選んでください。');};
    li.append(img,name,remove);$('photos').append(li);
  });
}
async function load(){
  ready=false;$('retry').hidden=true;render();status('部屋の情報を読み込んでいます…');
  try{
    if(!sceneId)throw Error('追加先の部屋が指定されていません。部屋一覧から開き直してください。');
    const [state,report]=await Promise.all([read('/viewer/state/'+encodeURIComponent(sceneId)),read('/scenes/'+encodeURIComponent(sceneId)+'/capture-quality')]);
    scene=state.scene;quality=report;
    if(!scene?.id||String(scene.id)!==sceneId)throw Error('追加先の部屋を確認できません。');
    $('roomName').textContent=scene.name||'部屋';
    const back=safeLocalPath(quality.result_url,location.origin);
    if(back){$('back').href=back;$('back').textContent='← 部屋に戻る';}
    $('qualityLink').href='/room/new?scene='+encodeURIComponent(sceneId);
    if(scene.reconstruction_model==='calibrated-rgbd')throw Error('この部屋は深度データから作られているため、この画面では写真を追加できません。');
    captureOptions(quality.options||{});
    const params=new URLSearchParams(location.search),key=params.get('target');
    if(key){
      const run=params.get('run'),result=params.get('result');
      if(![run,result].every(v=>/^[a-zA-Z0-9_-]{1,120}$/.test(v||'')))throw Error('撮影対象を確認できません。元の部屋から選び直してください。');
      const base='/static/generation-reviews/'+run+'/'+result+'/';
      const model=await read(base+'room.json');
      if(model.transform?.scene_id!==sceneId)throw Error('撮影対象と部屋が一致しません。元の部屋から選び直してください。');
      const evidence=await read(base+'quality.json').catch(()=>({}));
      const target=captureTargets(model,evidence).find(t=>t.key===key);
      if(!target)throw Error('この撮影対象は見つかりません。元の部屋から選び直してください。');
      captureTarget={run,result,key};
      $('generalPhotoTips').hidden=true;$('targetContext').hidden=false;$('targetName').textContent='撮影する場所：'+target.title;
      $('targetInstruction').textContent=target.instruction+' 横向きで、少し位置を変えて2〜4枚撮影してください。';$('targetWarning').textContent=target.note;
      const modelUrl='/static/solid-room.html?'+new URLSearchParams({run,result,capture:'1',target:key});
      $('targetModel').href=modelUrl;$('back').href=modelUrl;$('back').textContent='← 撮る場所に戻る';
    }
    const advice=photoAdvice(quality);$('suggestions').replaceChildren();
    for(const tip of advice.tips){const li=document.createElement('li');li.textContent=tip;$('suggestions').append(li);}
    $('repairNote').hidden=!advice.needsRepair;$('advice').hidden=!advice.tips.length&&!advice.needsRepair;
    ready=true;status('写真を撮るか、選んでください。');
  }catch(e){status(e.message,true);$('roomName').textContent=scene?.name||'部屋を確認できません';$('retry').hidden=!sceneId;}
  render();
}
function addFiles(input){
  try{
    const additions=Array.from(input.files||[]);if(!additions.length)return;
    const existing=new Set(photos.map(({file:f})=>[f.name,f.size,f.lastModified].join('|')));
    const unique=additions.filter(f=>{const key=[f.name,f.size,f.lastModified].join('|');if(existing.has(key))return false;existing.add(key);return true;});
    validateCaptureFiles({photos:[...photos.map(p=>p.file),...unique],parent_scene_id:sceneId});
    for(const file of unique)photos.push({file,url:URL.createObjectURL(file)});
    render();status(unique.length?photos.length+'枚を追加できます。':'同じ写真はすでに選択されています。');
  }catch(e){status(e.message+' JPEG・PNG・WebPの写真を選んでください。',true);}finally{input.value='';}
}
$('takePhoto').onclick=()=>$('cameraInput').click();$('choosePhotos').onclick=()=>$('photosInput').click();
for(const id of ['cameraInput','photosInput'])$(id).onchange=()=>addFiles($(id));
$('retry').onclick=load;
$('photoForm').onsubmit=async event=>{
  event.preventDefault();if(busy||!ready)return;
  let body;try{body=revisionBody({scene,quality,captureTarget,photos:photos.map(p=>p.file)});}catch(e){status(e.message,true);return;}
  busy=true;render();$('checkJobs').hidden=true;status('写真を送信しています。この画面を開いたままお待ちください。');
  let response;
  try{response=await fetch('/capture-jobs',{method:'POST',body});}
  catch{status('通信が途切れ、受付を確認できませんでした。写真はこの画面に残っています。再送信する前に、別のタブで部屋一覧の進み具合を確認してください。',true);$('checkJobs').hidden=false;busy=false;render();return;}
  try{
    const data=await response.json().catch(()=>null);
    if(!response.ok)throw Error(errorMessage(data,'写真を追加できませんでした。少し待ってから再度お試しください。'));
    if(!data?.id)throw Error('受付番号を確認できません。再送信する前に、別のタブで部屋一覧を確認してください。');
    accepted=true;location.assign('/room/new?job='+encodeURIComponent(data.id));
  }catch(e){status(e.message,true);$('checkJobs').hidden=false;busy=false;render();}
};
window.addEventListener('beforeunload',event=>{if(!accepted&&(busy||photos.length)){event.preventDefault();event.returnValue='';}});
load();
