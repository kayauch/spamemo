import {captureOptions,validateCaptureFiles,FURNITURE_LABELS} from './capture-utils.mjs?v=repair-capture2';

// Only photo-related evidence becomes shooting advice; geometry issues remain separate.
export function photoAdvice(quality={}){
  const recommendations=quality.recommendations||[];
  const photoCodes=new Set(['missing_photo_coverage','semantic_photo_support_missing','full_object_photo','side_or_back_photo','storage_side_depth_unresolved']);
  const repairCodes=new Set(['projection_depth_mismatch','furniture_mask_occlusion','measurement_conflict','confirm_category']);
  const visible=new Map((quality.objects||[]).filter(o=>!o.hidden).map(o=>[String(o.id),o]));
  const active=recommendations.filter(r=>r.object_id==null||visible.has(String(r.object_id)));
  const photo=active.filter(r=>photoCodes.has(r.code));
  const tips=[];
  if(photo.some(r=>r.kind==='wall'))tips.push('壁：家具の周囲に見えている壁と、部屋の角を一緒に撮影。');
  if(photo.some(r=>r.kind==='ceiling'))tips.push('天井：壁との境目と角が入るように撮影。');
  if(photo.some(r=>r.kind==='floor'))tips.push('床：壁との境目と、近くの家具の足元を一緒に撮影。');
  const labels=[...new Set(photo.filter(r=>r.object_id!=null).map(r=>FURNITURE_LABELS[visible.get(String(r.object_id))?.label]||'家具'))];
  if(labels.length)tips.push(`${labels.slice(0,3).join('・')}${labels.length>3?'など':''}：全体と、まだ見えていない側面を撮影。`);
  return {tips,needsRepair:active.some(r=>repairCodes.has(r.code))};
}

export function revisionBody({scene,quality,photos,captureTarget=null}){
  if(!scene?.id)throw Error('追加先の部屋を確認できません。部屋一覧から開き直してください。');
  if(scene.reconstruction_model==='calibrated-rgbd')throw Error('この部屋は深度データから作られているため、この画面では写真を追加できません。');
  validateCaptureFiles({photos,parent_scene_id:scene.id});
  const settings=captureOptions(quality?.options||{});
  const body=new FormData();
  body.append('parent_scene_id',scene.id);
  body.append('name',scene.name||'部屋');
  body.append('options',JSON.stringify(settings));
  if(captureTarget)body.append('capture_target',JSON.stringify(captureTarget));
  for(const photo of photos)body.append('photos',photo,photo.name);
  return body;
}
