const scene=new URLSearchParams(location.search).get('scene'),button=document.getElementById('solidRoomStart'),status=document.getElementById('solidRoomStatus');
let timer=null;
const retry=document.createElement('button');retry.textContent='歩ける部屋モデルを作り直す';retry.hidden=true;button.after(retry);
const previous=document.createElement('a');previous.textContent='前の結果を見る';previous.hidden=true;retry.after(previous);
const improve=document.createElement('a');improve.textContent='撮影を追加・家具と寸法を確認';improve.className='text-link';improve.href='/room/new'+(scene?'?scene='+encodeURIComponent(scene):'');previous.after(improve);
const preview=document.getElementById('solidRoomOrientationPreview');if(preview&&scene)preview.src=`/scenes/${encodeURIComponent(scene)}/frames/0`;
const endpoint=`/scenes/${encodeURIComponent(scene||'demo-room')}/solid-room-job`;
function render(data){
  retry.hidden=data.state==='running'||(!data.result_url&&!data.previous_result_url);retry.disabled=data.state==='running';
  previous.hidden=!data.previous_result_url;if(data.previous_result_url)previous.href=data.previous_result_url;
  document.getElementById('solidRoomUp').disabled=data.state==='running';
  if(data.up_axis)document.getElementById('solidRoomUp').value=data.up_axis;
  status.textContent=data.message||'この部屋の動画を使って処理します。';button.disabled=data.state==='running';
  button.textContent=data.state==='running'?'部屋モデルを作成中…':data.state==='complete'?(data.pipeline_version>=5?'作成済みの結果を確認':'家具と写真の分離を確認して作り直す'):'写真を貼った歩ける部屋を作る';
  for(const [id,key] of [['solidRoomResult','result_url'],['solidRoomReview','review_url']]){const link=document.getElementById(id);link.hidden=!data[key];if(data[key])link.href=data[key];}
  const completed=document.getElementById('completedRoomLink');
  if(completed){completed.hidden=!data.result_url;if(data.result_url)completed.href=data.result_url;}
  if(data.state==='running'){clearTimeout(timer);timer=setTimeout(()=>request('GET'),2500);}
}
async function request(method,force=false){
  clearTimeout(timer);button.disabled=true;
  try{const url=method==='POST'?endpoint+'?up_axis='+encodeURIComponent(document.getElementById('solidRoomUp').value)+(force?'&force=true':''):endpoint;const response=await fetch(url,{method});const data=await response.json();if(!response.ok)throw Error(data.detail||'処理状況を取得できませんでした');render(data);}
  catch(error){status.textContent=error.message;button.disabled=false;}
}
button.onclick=()=>request('POST');request('GET');
retry.onclick=()=>request('POST',true);
