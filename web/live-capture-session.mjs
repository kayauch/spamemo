import {revisionBody} from './photo-revision.mjs?v=repair-capture2';
import {shotKey} from './repair-capture-plan.mjs?v=repair-capture2';
export class LiveCaptureSession{
  constructor(targets){this.targets=[...new Map(targets.map(t=>[shotKey(t),t])).values()];this.index=0;this.shots=new Map();this.retake=false;}
  get current(){return this.targets[this.index]||null;}
  get photos(){return this.targets.filter(t=>this.shots.has(shotKey(t))).map(t=>this.shots.get(shotKey(t)));}
  get pending(){return this.targets.filter(t=>!this.shots.has(shotKey(t)));}
  nextPending(){const next=this.targets.findIndex((t,index)=>index>this.index&&!this.shots.has(shotKey(t)));return next<0?this.targets.length:next;}
  commit(file){
    const target=this.current;if(!target||!file?.size)throw Error('写真を保存できませんでした。もう一度撮影してください。');
    const previous=this.shots.get(shotKey(target));this.shots.set(shotKey(target),{file,target});
    this.index=this.retake?this.targets.length:this.nextPending();this.retake=false;return previous;
  }
  skip(){if(this.current){this.index=this.retake?this.targets.length:this.nextPending();this.retake=false;}}
  retry(key){const index=this.targets.findIndex(t=>shotKey(t)===key);if(index<0)throw Error('撮影対象が見つかりません');this.index=index;this.retake=true;}
  resume(){this.index=this.targets.findIndex(t=>!this.shots.has(shotKey(t)));if(this.index<0)this.index=this.targets.length;this.retake=false;}
  addTargets(targets){for(const target of targets)if(!this.targets.some(t=>shotKey(t)===shotKey(target)))this.targets.push(target);}
}
export function guidedRevisionBody({scene,quality,session,run,result}){
  const photos=session.photos;
  const body=revisionBody({scene,quality,photos:photos.map(p=>p.file)});
  body.append('capture_targets',JSON.stringify(photos.map(p=>({run,result,key:p.target.key}))));return body;
}
