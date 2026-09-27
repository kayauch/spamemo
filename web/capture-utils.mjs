export const FURNITURE_LABELS={bed:'ベッド',locker:'ロッカー',shelf:'棚',box:'収納箱',chair:'椅子',sofa:'ソファ',desk:'デスク',table:'テーブル',wardrobe:'衣類収納',cabinet:'キャビネット',bookshelf:'本棚',curtain:'カーテン',coffee_table:'ローテーブル',dining_chair:'ダイニングチェア'};
export function numberInRange(value,min,max,label,{optional=false,integer=false}={}){
  if(String(value??'').trim()===''){if(optional)return null;throw Error(`${label}を入力してください。`);}
  const n=Number(value);if(!Number.isFinite(n)||n<min||n>max||(integer&&!Number.isInteger(n)))throw Error(`${label}は${min}〜${max}${integer?'の整数':''}で入力してください。`);return n;
}
export function captureOptions(values){
  if(values.room_type!=null&&!['research','bedroom','storage','other'].includes(values.room_type))throw Error('部屋のタイプを確認してください。');
  if(!['image-top','image-left','image-right','image-bottom'].includes(values.up_axis))throw Error('画像の向きを確認してください。');
  if(!['high','standard'].includes(values.texture_quality))throw Error('写真の画質を確認してください。');
  return {room_type:values.room_type??null,up_axis:values.up_axis,ceiling_height_m:numberInRange(values.ceiling_height_m,1.5,10,'天井高',{optional:true}),reference_length_m:numberInRange(values.reference_length_m,.3,50,'最も長い壁の長さ',{optional:true}),reference_axis:'longest_wall',max_frames:numberInRange(values.max_frames,12,120,'画像の上限',{integer:true}),furniture_frame_budget:numberInRange(values.furniture_frame_budget,8,32,'家具の確認枚数',{integer:true}),texture_quality:values.texture_quality,refine_cameras:Boolean(values.refine_cameras)};
}
export function validateCaptureFiles({videos=[],photos=[],panoramas=[],depth_bundle=[],parent_scene_id=''}){
  const groups=[['動画',videos,6,256,['mp4','mov','avi','mkv','webm','m4v']],['写真',photos,64,64,['jpg','jpeg','png','webp']],['パノラマ',panoramas,4,64,['jpg','jpeg','png','webp']],['深度データ',depth_bundle,1,256,['zip']]];
  let total=0,count=0;
  for(const [label,files,limit,megabytes,extensions] of groups){if(files.length>limit)throw Error(`${label}は最大${limit}個まで選んでください。`);for(const file of files){count++;if(!extensions.includes(String(file.name).split('.').pop().toLowerCase()))throw Error(`${label}のファイル形式に対応していません。`);if(!Number.isFinite(file.size)||file.size<=0||file.size>megabytes*1024**2)throw Error(`${label}は空でない${megabytes}MB以下のファイルを使ってください。`);total+=file.size;}}
  if(!count)throw Error('動画・写真・パノラマのいずれかを選んでください。');
  if(total>512*1024**2)throw Error('一度に送るファイルは合計512MB以下にしてください。');
  if(depth_bundle.length&&(parent_scene_id||videos.length||photos.length||panoramas.length))throw Error('深度データは単独で、新しい部屋として取り込んでください。');
}
export function validatePanoramaDimensions(width,height){if(!Number.isFinite(width)||!Number.isFinite(height)||width<512||height<256||Math.abs(width/height-2)>.02)throw Error('パノラマには横2：縦1の全天球画像（512×256以上）を使ってください。');}
export function furnitureOverride({id,label,hidden,dimensions}){
  if(!Object.hasOwn(FURNITURE_LABELS,label))throw Error('家具の種類を確認してください。');
  if(!Array.isArray(dimensions)||dimensions.length!==3)throw Error('家具の幅・奥行・高さを確認してください。');
  const result={id:String(id),label,hidden:Boolean(hidden)};
  const entered=dimensions.map(value=>String(value??'').trim());
  if(entered.some(Boolean)){if(!entered.every(Boolean))throw Error('家具の幅・奥行・高さは3つとも入力してください。');result.dimensions_m=entered.map(value=>numberInRange(value,.03,10,'家具の寸法'));}
  return result;
}
export function safeLocalPath(value,origin){if(typeof value!=='string'||!value)return null;try{const url=new URL(value,origin);return url.origin===origin&&['http:','https:'].includes(url.protocol)?url.pathname+url.search+url.hash:null;}catch{return null;}}
export function errorMessage(data,fallback){if(typeof data?.detail==='string')return data.detail;if(Array.isArray(data?.detail))return data.detail.map(x=>x.msg||'入力を確認してください').join(' / ');return fallback;}
export function settingsChanged(settings,previous={}){return Object.entries(settings).some(([key,value])=>key==='furniture_overrides'?value.length>0:value!==(previous[key]??(value===null?null:undefined)));}
