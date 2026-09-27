import {FURNITURE_LABELS} from './capture-utils.mjs?v=4';
const photoCodes=new Set(['missing_photo_coverage','semantic_photo_support_missing','full_object_photo','side_or_back_photo','storage_side_depth_unresolved','unobserved_structure']);
const repairCodes=new Set(['projection_depth_mismatch','furniture_mask_occlusion','measurement_conflict','confirm_category']);
export function captureTargets(room,quality={}){
  const groups=new Map(),counts={};
  for(const surface of room.surfaces||[]){
    const object=surface.object_id!=null,kind=object?'object':surface.kind;
    if(!object&&!['wall','floor','ceiling'].includes(kind))continue;
    const key=object?'object:'+surface.object_id:'surface:'+surface.id;
    if(!groups.has(key))groups.set(key,{key,kind,object_id:object?String(surface.object_id):null,label:surface.object_label,shape:object?null:{origin:surface.origin,u:surface.u,v:surface.v},surface_ids:[]});
    groups.get(key).surface_ids.push(surface.id);
  }
  for(const object of room.furniture||[]){
    if(object.hidden)continue;
    const key='object:'+object.id;
    if(!groups.has(key))groups.set(key,{key,kind:'object',object_id:String(object.id),label:object.label,surface_ids:[]});
  }
  const hidden=new Set((room.furniture||[]).filter(o=>o.hidden).map(o=>String(o.id)));
  return [...groups.values()].filter(t=>!hidden.has(t.object_id)).map(t=>{
    const word=t.kind==='object'?(FURNITURE_LABELS[t.label]||'家具'):{wall:'壁',floor:'床',ceiling:'天井'}[t.kind];
    counts[word]=(counts[word]||0)+1;t.title=(t.kind==='object'&&room.object_names?.[t.object_id])||word+(['wall','object'].includes(t.kind)?' '+counts[word]:'');
    const rows=(quality.recommendations||[]).filter(r=>t.object_id!=null?String(r.object_id)===t.object_id:t.surface_ids.includes(r.surface_id));
    t.repair=rows.some(r=>repairCodes.has(r.code));t.photo=rows.some(r=>photoCodes.has(r.code));
    t.score=Math.max(0,...rows.map(r=>Number(r.missing_fraction)||0));
    const faceReports=t.surface_ids.map(id=>(quality.surfaces||[]).find(s=>s.surface_id===id)||(room.surfaces||[]).find(s=>s.id===id));
    t.coverageKnown=faceReports.length>0&&faceReports.every(s=>Number.isFinite(s?.supported_fraction));
    t.supportedFraction=t.coverageKnown?Math.min(...faceReports.map(s=>s.supported_fraction)):null;
    t.candidate=t.photo&&!t.repair;
    t.frames=[...new Set((quality.surfaces||[]).filter(s=>t.surface_ids.includes(s.surface_id)).flatMap(s=>s.frames_used||[]))].filter(n=>Number.isInteger(n)&&n>=0);
    t.instruction=t.kind==='floor'?'床と壁の境目、近くの家具の足元を一緒に入れてください。':t.kind==='ceiling'?'天井と壁の境目、部屋の角を一緒に入れてください。':t.kind==='wall'?'青い壁の全体と隣の角を。家具の周囲に見えている壁も入れてください。':'家具の全体と側面を。背後の壁との境目も入れてください。';
    t.note=t.repair?'形状や位置のずれの可能性があります。元写真と見比べてください。写真追加だけでは直らない場合があります。':t.photo?'使える写真の根拠が少ない候補です。実際に不足しているか確認して撮影してください。':'自分で選んだ対象です。見た目が気になる部分を撮影してください。';
    return t;
  });
}
export function targetPhotoUrl(sceneId,run,result,key){return '/static/add-room-photos.html?'+new URLSearchParams({scene:sceneId,run,result,target:key});}
