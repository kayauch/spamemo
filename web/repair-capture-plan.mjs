// Plan overlapping viewpoints, not a claim that a photograph matches its target.
export const MAX_REPAIR_PHOTOS=64;
export const shotKey=target=>target.shotId||target.key;

export function repairCapturePlan(targets,{selectedKey=null}={}){
  const needed=targets.filter(t=>t.key===selectedKey||t.photo||!t.coverageKnown||t.supportedFraction<.9);
  needed.sort((a,b)=>(a.key===selectedKey?-1:b.key===selectedKey?1:0));
  const shots=[];
  for(const target of needed){
    if(target.kind==='object'){
      const views=[
        ['front','全体','家具全体と周りの壁・床が入る位置まで下がってください。'],
        ['left','左から斜め','家具は動かさず、少し左へ移動。正面と見える側面を一緒に入れてください。'],
        ['right','右から斜め','家具は動かさず、少し右へ移動。正面と見える側面を一緒に入れてください。']
      ];
      views.forEach(([id,label,instruction],index)=>shots.push({...target,shotId:target.key+'/'+id,targetTitle:target.title,title:target.title+' · '+label,shotIndex:index,shotCount:views.length,instruction:'横向きで、'+instruction+' 同じ対象の写真と重なる特徴を入れ、止まって撮影してください。近づけない場所はスキップできます。'}));
      continue;
    }
    const shape=target.shape||{},u=Math.hypot(...(shape.u||[1,0,0])),v=Math.hypot(...(shape.v||[0,1,0]));
    const ratio=u>0&&v>0?u/v:1;
    // A broad surface is split further; every cell keeps overlap with its neighbours.
    const columns=Math.max(2,Math.ceil(ratio)),rows=target.kind==='wall'?2:Math.max(2,Math.ceil(1/ratio));
    for(let row=0;row<rows;row++)for(let column=0;column<columns;column++){
      const index=row*columns+column,overlap=.12;
      const region={u0:Math.max(0,(column-overlap)/columns),u1:Math.min(1,(column+1+overlap)/columns),v0:Math.max(0,(row-overlap)/rows),v1:Math.min(1,(row+1+overlap)/rows)};
      const context=target.kind==='ceiling'?'壁との境目や照明を一緒に。無理に背を反らさず、足元に気をつけてください。':target.kind==='floor'?'床と壁の境目、近くの家具の足元を一緒に入れてください。':'家具を壁として撮らず、家具の周りに見える壁と部屋の角を入れてください。';
      shots.push({...target,shotId:target.key+'/cell-'+row+'-'+column,targetTitle:target.title,title:target.title+' · 区画 '+(index+1)+'/'+(columns*rows),shotIndex:index,shotCount:columns*rows,region,instruction:'横向きで、3Dの青い区画を周囲も含めて撮影。'+context+' 前の写真と重なる部分を残してください。'});
    }
  }
  // Cover all objects at least once before spending the remaining budget on angles.
  return shots.sort((a,b)=>Number(a.kind==='object'&&a.shotIndex>0)-Number(b.kind==='object'&&b.shotIndex>0));
}

export function photoOrientation({landscape,width,height}){
  if(!landscape)return {allowed:false,reason:'portrait',message:'スマホを横向きにしてください。横向きになると撮影できます。'};
  if(!(width>0&&height>0))return {allowed:false,reason:'loading',message:'カメラの映像を確認しています。'};
  if(width<=height)return {allowed:false,reason:'portrait-frame',message:'映像が縦向きです。横向きのままカメラを開き直してください。'};
  return {allowed:true,reason:'landscape',message:''};
}
