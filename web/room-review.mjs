import * as THREE from 'three';
import {wallBetween,insideFootprint,placementIssues} from './room-review-math.mjs';
export function installRoomReview({data,surfaces,objectMeshes,camera,orbit,scene,walk,stopWalking,overview}){
 const panel=document.querySelector('#viewerSettings .settings-section');
 const label=document.createElement('label'),toggle=document.createElement('input');toggle.type='checkbox';toggle.checked=true;label.append(toggle,'手前の壁を自動で透過');panel.prepend(label);
 const walls=surfaces.filter(m=>m.userData.kind==='wall'),original=new Map(walls.map(m=>[m,{opacity:m.material.opacity,transparent:m.material.transparent,depthWrite:m.material.depthWrite}]));
 const marks=new THREE.Group();scene.add(marks);const boxes=[];
 for(const [id,item] of objectMeshes){const box=new THREE.Box3();for(const m of item.meshes)box.expandByObject(m);if(!box.isEmpty())boxes.push({id,label:item.label,min:box.min.toArray(),max:box.max.toArray()});}
 const issues=placementIssues(boxes,data.floor,data.ceiling-data.floor);
 const details=document.createElement('details'),summary=document.createElement('summary'),note=document.createElement('p');summary.textContent=`配置を確認${issues.length?'（'+issues.length+'件の候補）':''}`;note.className='dialog-note';note.textContent='推定した家具の範囲から、浮き・床への埋まり・大きな重なりを確認します。枠の誤差や見えない支えもあるため、候補を見て判断してください。配置は自動では変更しません。';details.append(summary,note);panel.append(details);
 const names={bed:'ベッド',sofa:'ソファ',cabinet:'収納',wardrobe:'収納',locker:'ロッカー',shelf:'棚',desk:'机',chair:'椅子',dining_chair:'椅子',box:'箱',coffee_table:'テーブル'};
 const labels={floating:'支えから離れている可能性',below_floor:'床に埋まっている可能性',overlap:'家具が重なっている可能性'};
 function clear(){for(const child of [...marks.children]){child.geometry.dispose();child.material.dispose();marks.remove(child);}}
 if(!issues.length){const empty=document.createElement('p');empty.className='dialog-note';empty.textContent='検査できる家具に大きな問題候補はありません。小物や複雑な形の重なりは目視でも確認してください。';details.append(empty);}
 for(const issue of issues){const button=document.createElement('button');button.type='button';button.className='room-secondary';button.style.cssText='display:block;width:100%;text-align:left;white-space:normal;margin:8px 0';button.textContent=issue.ids.map(id=>objectMeshes.get(id)?.name||`${names[objectMeshes.get(id)?.label]||objectMeshes.get(id)?.label} ${id}`).join(' / ')+'：'+labels[issue.kind];button.onclick=()=>{
  stopWalking();overview();clear();const total=new THREE.Box3();
  for(const id of issue.ids){const b=boxes.find(b=>b.id===id),box=new THREE.Box3(new THREE.Vector3(...b.min),new THREE.Vector3(...b.max));total.union(box);const helper=new THREE.Box3Helper(box,0xffbb55);helper.material.depthTest=false;helper.material.transparent=true;helper.material.opacity=.85;helper.renderOrder=20;marks.add(helper);}
  const target=total.getCenter(new THREE.Vector3()),distance=Math.max(1,total.getSize(new THREE.Vector3()).length()*1.5),direction=camera.position.clone().sub(orbit.target).normalize();orbit.target.copy(target);camera.position.copy(target).addScaledVector(direction,distance);camera.position.z=Math.max(camera.position.z,data.ceiling+.3);orbit.update();document.getElementById('viewerSettings').close();
 };details.append(button);}
 const clearButton=document.createElement('button');clearButton.type='button';clearButton.className='room-secondary';clearButton.textContent='確認の枠を消す';clearButton.onclick=clear;details.append(clearButton);
 let last=0;
 function tick(now){if(now-last<50)return;last=now;
  const p=camera.position.toArray(),target=orbit.target.toArray();
  const inRoom=insideFootprint(p,data.footprint||data.navigation?.footprint)&&p[2]<=data.ceiling&&p[2]>=data.floor;
  const editing=document.body.dataset.captureGuide==='true'||document.querySelector('.wall-photo[open]');
  for(const mesh of walls){const mat=mesh.material,base=original.get(mesh),fade=toggle.checked&&!walk.active&&!inRoom&&!editing&&wallBetween(p,target,mesh.userData);
   const opacity=fade ? 0.14 : base.opacity;
   if(mat.opacity!==opacity){mat.opacity=opacity;mat.transparent=fade||base.transparent;mat.depthWrite=fade?false:base.depthWrite;mat.needsUpdate=true;}
  }
 }
 return {tick};
}
