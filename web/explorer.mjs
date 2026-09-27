import {icon} from './spamemo-shell.mjs';
export function installExplorer({camera,orbit,renderer,finder,overview,startWalking,getViewMode,canWalk}){
 const $=id=>document.getElementById(id),stage=$('stage');document.body.classList.add('explorer');
 const css=document.createElement('link');css.rel='stylesheet';css.href='/static/explorer.css?v=repair-capture2';document.head.append(css);
 document.querySelector('.minimal-move')?.remove();
 const nav=document.createElement('nav');nav.className='explorer-nav';nav.setAttribute('aria-label','部屋の操作');
 function label(node,name,glyph){node.replaceChildren(icon(glyph),Object.assign(document.createElement('span'),{textContent:name}));node.setAttribute('aria-label',name);return node;}
 const repair=label(document.createElement('button'),'部屋の修正','settings');repair.type='button';repair.id='exploreRepair';repair.className='explorer-repair';repair.setAttribute('aria-controls','captureGuide');repair.title='足りない写真を案内に沿って撮影し、部屋を修正';repair.querySelector('path').setAttribute('d','m16 3 5 5-12 12-6 1 1-6ZM14 5l5 5');repair.onclick=()=>{if(document.body.dataset.finderState==='guiding')$('stopGuide').click();document.dispatchEvent(new Event('repair-room'));};
 const move=document.createElement('button');move.type='button';move.className='explorer-move';let shownMode='';
 function syncMode(){
  const mode=getViewMode(),available=mode==='walk'||canWalk();move.disabled=!available;
  if(shownMode===mode)return;shownMode=mode;const walking=mode==='walk';
  label(move,walking?'俯瞰へ':'移動へ',walking?'map':'chevron');
  if(!walking)move.querySelector('path').setAttribute('d','M21 3 3 10l8 3 3 8Z');
  const status=document.createElement('small');status.className='explorer-mode-status';status.textContent=walking?'移動モード':'俯瞰モード';move.append(status);
  move.setAttribute('aria-label',(walking?'俯瞰':'移動')+'モードに切り替え（現在：'+(walking?'移動':'俯瞰')+'モード）');
  move.title=available?move.getAttribute('aria-label'):'この部屋には歩ける範囲の情報がありません';
  document.body.dataset.viewMode=mode;
  if(typeof hint!=='undefined')hint.textContent=walking?'スワイプで見回す · 矢印で移動':'スワイプで回転 · ピンチで拡大';
 }
 move.onclick=()=>{fit.disconnect();if(getViewMode()==='walk'){if(document.body.dataset.finderState==='guiding')$('stopGuide').click();overview();}else startWalking();syncMode();};
 const capture=label(document.createElement('button'),'撮影','camera');capture.querySelector('path').setAttribute('d','M8 3H4v5m12-5h4v5M4 16v5h4m8 0h4v-5');capture.type='button';capture.className='explorer-capture';capture.onclick=()=>document.dispatchEvent(new Event('storage-photograph'));
 const storage=document.querySelector('.shelf-launch');label(storage,'登録','box');
 const map=$('openMap');label(map,'マップ','map');const find=label(document.createElement('button'),'探す','search');find.type='button';find.className='explorer-find';find.onclick=()=>{$('roomSearch').hidden=false;$('findQuery').focus();};nav.append(repair,move,find,storage,map);capture.className="explorer-search";document.querySelector(".header-actions").prepend(capture);stage.append(nav);
 const mini=document.createElement('button');mini.type='button';mini.className='explorer-mini';mini.setAttribute('aria-label','ミニマップを開く');mini.append(Object.assign(document.createElement('span'),{textContent:'ミニマップ'}));const canvas=document.createElement('canvas');canvas.width=240;canvas.height=200;mini.append(canvas);mini.onclick=()=>map.click();stage.append(mini);
 const zoom=document.createElement('div');zoom.className='explorer-zoom';zoom.setAttribute('aria-label','拡大・縮小');for(const [text,name,delta] of [['＋','拡大',-8],['−','縮小',8]]){const b=document.createElement('button');b.type='button';b.textContent=text;b.setAttribute('aria-label',name);b.onclick=()=>{camera.fov=Math.max(30,Math.min(95,camera.fov+delta));camera.updateProjectionMatrix();};zoom.append(b);}stage.append(zoom);
 const hint=document.createElement('p');hint.className='explorer-hint';hint.textContent='スワイプで見回す · ピンチで拡大';stage.append(hint);
 document.body.dataset.explorerSearch='true';
 $('roomSearch').hidden=false;
 const quick=document.createElement('button');quick.type='button';quick.className='explorer-quick-search';quick.textContent='充電ケーブルどこ？';quick.disabled=$('findQuery').disabled;
 quick.onclick=()=>{if($('findQuery').disabled)return;$('findQuery').value='充電ケーブル';$('roomSearch').hidden=false;$('roomSearch').requestSubmit();};stage.append(quick);


 document.body.dataset.minimalWalk='true';overview();const offset=camera.position.clone().sub(orbit.target);const fit=new ResizeObserver(()=>{if(document.body.dataset.captureGuide==='true'||getViewMode()==='walk'){fit.disconnect();return;}camera.position.copy(orbit.target).addScaledVector(offset,Math.max(1,.85/(stage.clientWidth/stage.clientHeight)));orbit.update();});fit.observe(stage);stage.addEventListener('pointerdown',()=>fit.disconnect(),{once:true});let last=0;syncMode();
 return {tick(now){syncMode();if(now-last<200)return;last=now;if(!finder.drawMap)return;finder.drawMap();canvas.getContext('2d').drawImage($('roomMap'),0,0,canvas.width,canvas.height);}};
}
