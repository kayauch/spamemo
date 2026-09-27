export function generationPhase(job={}) {
  const failed=['failed','interrupted'].includes(job.state);
  if(job.state==='complete')return {index:4,percent:100,title:'3D空間が完成しました！',detail:'撮影した部屋を、見回せる3D空間にしました。',failed:false};
  const step=Number(/^(\d)\/6/.exec(job.message||'')?.[1]||0);
  const index=step>=5?3:step===4?2:step>=1?1:0;
  const percent=step?[0,40,52,64,76,86,94][step]||15:15;
  const titles=['動画を解析しています…','部屋の形を作成しています…','家具・物を認識しています…','3D空間を仕上げています…'];
  const details=['撮影した動画から、部屋の形を調べています。','壁・床・天井の形を復元しています。','複数の写真から、家具の位置を確認しています。','写真を貼り、部屋の中を歩けるようにしています。'];
  return {index,percent,title:failed?'処理が止まりました':titles[index],detail:failed?'撮影データは保存されています。内容を確認して再開できます。':details[index],failed};
}

export function installGenerationProgress() {
  const $=id=>document.getElementById(id);let preview,previewModule,lastJob;
  function render(job) {
    lastJob=job;const p=generationPhase(job),complete=p.index===4;
    $('generationCard').dataset.phase=complete?'complete':p.failed?'failed':String(p.index);
    $('generationTitle').textContent=p.title;$('generationDetail').textContent=p.detail;
    $('generationPercent').textContent=p.failed?'!':p.percent+'%';
    $('generationRing').style.setProperty('--progress',p.percent+'%');
    $('generationBar').style.width=p.percent+'%';$('generationBarLabel').textContent=p.percent+'%';
    $('generationMeter').setAttribute('aria-valuenow',String(p.percent));
    $('generationMeter').setAttribute('aria-valuetext',p.failed?'処理が停止しています':p.percent+'%（処理段階に基づく目安）');
    $('generationMeter').hidden=complete||p.failed;
    $('generationSteps').hidden=complete;
    $('generationRestart').hidden=!complete;
    const photoLink=$('generationAddPhotos'),sceneId=job.scene_id||job.scene;
    if(photoLink){photoLink.hidden=!complete||!sceneId;if(sceneId)photoLink.href='/static/add-room-photos.html?scene='+encodeURIComponent(sceneId);}
    $('generationQualityToggle').hidden=!complete;
    $('editCapture').hidden=!p.failed;
    document.querySelectorAll('#generationSteps li').forEach((li,i)=>{
      li.classList.toggle('done',i<p.index);li.classList.toggle('active',i===p.index&&!p.failed);
      li.querySelector('span').textContent=i<p.index?'✓':i===p.index?'●':'·';
      if(i===p.index)li.setAttribute('aria-current','step');else li.removeAttribute('aria-current');
    });
    const hasScene=!!(job.scene_id||job.scene);
    $('generationVisual').hidden=!hasScene;
    $('generationRing').hidden=hasScene||complete;
    if(hasScene){
      previewModule??=import('./generation-preview.mjs?v=20260926-experience2');
      previewModule.then(({GenerationPreview})=>{
        preview??=new GenerationPreview($('generationCanvas'),$('generationPreviewNote'));
        return preview.update(lastJob);
      }).catch(()=>{$('generationPreviewNote').textContent='プレビューを表示できません。処理は引き続き進みます。';});
    }
  }
  $('generationQualityToggle').onclick=()=>{
    const details=$('generationQualityDetails');details.hidden=!details.hidden;
    $('generationQualityToggle').setAttribute('aria-expanded',String(!details.hidden));
  };
  return {render,reconnecting(message){render(lastJob||{state:'running'});$('generationTitle').textContent='通信を再確認しています…';$('generationDetail').textContent=message;$('editCapture').hidden=true;},error(message){const p=generationPhase({...lastJob,state:'failed'});$('generationTitle').textContent=p.title;$('generationDetail').textContent=message;$('generationCard').dataset.phase='failed';$('generationMeter').hidden=true;$('generationPercent').textContent='!';$('editCapture').hidden=false;},submitting(message){render({state:'queued',stage:'capture'});$('result').hidden=true;$('generationTitle').textContent='撮影データを送信しています…';$('generationDetail').textContent=message;$('generationMeter').hidden=true;$('generationPercent').textContent='…';}};
}
