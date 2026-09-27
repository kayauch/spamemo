const $=id=>document.getElementById(id),query=new URLSearchParams(location.search);
const run=query.get('run'),result=query.get('result');
let source,sourceUrl,scene,timer,reportBase,references=[],showLines=true;
const labels={floor:'床',ceiling:'天井'};
const reasons={heldout_improved:'別の写真でも一致が改善 → 補正',already_aligned:'位置のずれが小さいため維持',insufficient_independent_views:'異なる位置からの根拠が足りず維持',heldout_views_disagree:'別の写真と一致しないため維持',inconsistent_depth:'奥行きの推定が不安定なため維持',unsafe_shared_corners:'部屋の形が崩れるため維持',invalid_room_height:'高さを確認できず維持',correction_too_large:'ずれが大きく自動補正できず維持'};
async function read(url,options){const r=await fetch(url,options);const data=await r.json();if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:'処理を開始できませんでした');return data;}
function image(){const ref=references[Number($('frame').value)];if(!ref)return;$('oldImage').src=new URL(showLines?ref.before:ref.photo,reportBase);$('newImage').src=new URL(showLines?ref.after:ref.photo,reportBase);}
function localUrl(value){const url=new URL(value,location.origin);if(url.origin!==location.origin||!url.pathname.startsWith('/static/'))throw Error('比較先のURLが不正です');return url;}
async function showResult(url){
  const target=localUrl(url),q=target.searchParams;
  if(![q.get('run'),q.get('result')].every(x=>/^[\w-]+$/.test(x||'')))throw Error('比較結果が不正です');
  const base=`/static/generation-reviews/${q.get('run')}/${q.get('result')}/`;
  const model=await read(base+'room.json');if(!model.photo_alignment)throw Error('位置合わせの結果が見つかりません');
  const reportUrl=new URL(model.photo_alignment.report_url,new URL(base,location.origin));localUrl(reportUrl.href);reportBase=reportUrl;
  const report=await read(reportUrl);references=report.references||[];
  $('result').href=target.href;$('result').hidden=false;
  if(report.before_result_url){$('before').href=localUrl(report.before_result_url).href;$('before').hidden=false;}
  const corrected=new Set(report.cameras.frames.filter(r=>r.accepted).map(r=>r.index));
  $('frame').replaceChildren(...references.map((r,i)=>new Option(`写真 ${r.frame+1}${corrected.has(r.frame)?'（カメラ補正あり）':''}`,String(i))));image();
  $('cameraChecks').textContent=corrected.size?`写真 ${[...corrected].map(i=>i+1).join('、')}：別の写真との一致が改善しました。それ以外の写真は根拠不足・改善なし・基準となる撮影位置などのため元の位置を維持しました。`:'改善を確認できるカメラ位置の補正はありませんでした。';
  $('summary').textContent=`カメラ ${report.cameras.accepted_count} 枚、床・壁・天井 ${report.geometry.accepted_count} 面の位置を補正しました。`;
  $('checks').replaceChildren(...report.geometry.surfaces.map(item=>{const tr=document.createElement('tr'),name=document.createElement('td'),state=document.createElement('td');name.textContent=labels[item.surface_id]||`壁 ${Number(item.surface_id.split('-')[1])+1}`;state.textContent=reasons[item.reason]||'補正せず維持';tr.append(name,state);return tr;}));
  const relevant=model.surfaces.filter(s=>s.kind==='floor'||s.kind==='ceiling');
  $('coverage').textContent='補正後に写真を貼れた範囲：'+relevant.map(s=>`${labels[s.kind]} ${Math.round(s.supported_fraction*100)}%`).join(' ／ ');
  $('status').textContent=report.cameras.accepted_count+report.geometry.accepted_count?'補正と再生成が完了しました。元写真と重ねて確認できます。':'写真との一致が改善する補正を確認できなかったため、位置を維持しました。';
  $('review').hidden=false;$('start').hidden=true;
}
async function poll(){
  clearTimeout(timer);
  try{
    const job=await read(`/scenes/${scene}/solid-room-job`);
    if(job.align_photos&&job.previous_result_url===sourceUrl){
      $('status').textContent=job.message||'写真と照合しています…';
      if(job.state==='running'){$('start').disabled=true;timer=setTimeout(poll,2500);return;}
      if(job.state==='complete'){await showResult(job.result_url);return;}
      if(job.state==='failed'||job.state==='interrupted')throw Error('補正処理を完了できませんでした。元の部屋は残っています。もう一度お試しください。');
    }
    if(job.state==='running')throw Error('別の部屋処理が進行中です。完了後にお試しください。');
    $('start').disabled=false;
  }catch(error){$('error').textContent=error.message;$('start').disabled=false;}
}
$('start').onclick=async()=>{clearTimeout(timer);$('start').disabled=true;$('error').textContent='';try{await read(`/scenes/${scene}/photo-alignment`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run,result})});await poll();}catch(error){$('error').textContent=error.message;$('start').disabled=false;}};
$('frame').onchange=image;$('lines').onclick=()=>{showLines=!showLines;$('lines').setAttribute('aria-pressed',String(showLines));image();};
try{
  if(![run,result].every(v=>/^[\w-]+$/.test(v||'')))throw Error('部屋からこの画面を開いてください');
  sourceUrl=`/static/solid-room.html?run=${run}&result=${result}`;$('back').href=sourceUrl;
  source=await read(`/static/generation-reviews/${run}/${result}/room.json`);scene=source.transform?.scene_id;
  if(!/^[0-9a-f-]{36}$/i.test(scene||''))throw Error('元の部屋が見つかりません');
  if(source.photo_alignment)await showResult(sourceUrl);
  else{$('status').textContent='保存済みの写真で補正できます。処理には数分かかります。';await poll();}
}catch(error){$('error').textContent=error.message;$('status').textContent='';$('start').disabled=true;}
window.addEventListener('pagehide',()=>clearTimeout(timer));
