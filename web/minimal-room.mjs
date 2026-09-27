export function installMinimalRoom(stopWalking){
 const css=document.createElement('link');css.rel='stylesheet';css.href='/static/minimal-room.css?v=1';document.head.append(css);
 const $=id=>document.getElementById(id);const move=document.createElement('button');move.className='minimal-move';move.textContent='移動';move.type='button';move.setAttribute('aria-pressed','false');move.setAttribute('aria-controls','roomMovement');move.onclick=()=>{const active=move.getAttribute('aria-pressed')!=='true';move.setAttribute('aria-pressed',String(active));document.body.dataset.minimalWalk=String(active);if(!active)stopWalking();};$('stage').append(move);
 const photos=$('addRoomPhotos');photos.addEventListener('click',()=>$('viewerSettings').close());$('viewerSettings').querySelector('.settings-links').prepend(photos);
 const settings=$('viewerSettings');for(const section of [...settings.querySelectorAll(':scope > .settings-section')]){const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent=section.querySelector('h3')?.textContent||'表示';section.querySelector('h3')?.remove();section.before(details);details.append(summary,section);}
}
