import {recordingMime,recordingFile,MAX_RECORDING_BYTES,MAX_RECORDING_SECONDS} from './video-capture-utils.mjs?v=20260926-ui2';

// Media lifecycle is separate from the screen so interruption/race cases can be tested.
export class CaptureRecorder {
  constructor({mediaDevices=globalThis.navigator?.mediaDevices,Recorder=globalThis.MediaRecorder,onState=()=>{},onClip=()=>{},onError=()=>{},now=()=>Date.now(),schedule=(fn,ms)=>globalThis.setInterval(fn,ms),unschedule=id=>globalThis.clearInterval(id)}={}){
    Object.assign(this,{mediaDevices,Recorder,onState,onClip,onError,now,schedule,unschedule});
    this.stream=null;this.recorder=null;this.generation=0;this.state='closed';this.timer=null;this.stopPromise=null;this.opening=false;
  }
  change(state,detail={}){this.state=state;this.onState(state,detail);}
  async open(){
    if(!this.mediaDevices?.getUserMedia||!this.Recorder)throw Error('このブラウザでは直接録画できません。「端末のカメラで撮影」か動画ファイルを選んでください。');
    if(this.stream||this.recorder||this.opening)await this.close();const generation=++this.generation;this.opening=true;this.change('opening',{seconds:0});
    try{const stream=await this.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1920},height:{ideal:1080}},audio:false});
      if(generation!==this.generation){stream.getTracks().forEach(track=>track.stop());return null;}
      this.stream=stream;this.opening=false;stream.getVideoTracks().forEach(track=>{track.onended=()=>{this.onError('カメラとの接続が終了しました。保存できた動画を確認してください。');this.close();};});this.change('ready');return stream;
    }catch(error){if(generation!==this.generation)return null;this.opening=false;this.change('closed');throw error;}
  }
  start(){
    if(!this.stream||this.state!=='ready')throw Error('カメラの準備ができていません。');
    const mime=recordingMime(this.Recorder);const recorder=new this.Recorder(this.stream,{...(mime?{mimeType:mime}:{}),videoBitsPerSecond:5_000_000});
    this.recorder=recorder;const chunks=[];let bytes=0,failed=false;const start=this.now();this.startedAt=start;
    this.stopPromise=new Promise(resolve=>{this.resolveStop=resolve;});
    recorder.ondataavailable=event=>{if(event.data?.size){chunks.push(event.data);bytes+=event.data.size;if(bytes>=MAX_RECORDING_BYTES)this.stop('ファイルサイズの上限に近づいたため録画を停止しました。');}};
    recorder.onerror=()=>{failed=true;this.onError('録画中にエラーが発生しました。もう一度撮影するか、端末のカメラを使ってください。');this.stop();};
    recorder.onstop=()=>{this.unschedule(this.timer);this.timer=null;const finish=this.resolveStop;this.resolveStop=null;this.recorder=null;
      if(!failed){try{const type=recorder.mimeType||mime||chunks[0]?.type||'video/webm';const file=recordingFile(new Blob(chunks,{type}),start);this.onClip(file,{seconds:(this.now()-start)/1000});}catch(error){this.onError(error.message);}}
      chunks.length=0;this.change(this.stream?'ready':'closed');finish?.();};
    try{recorder.start(1000);}catch(error){this.recorder=null;this.resolveStop?.();this.resolveStop=null;this.change('ready');throw error;}
    this.change('recording',{seconds:0});this.timer=this.schedule(()=>{const seconds=(this.now()-start)/1000;this.onState('recording',{seconds});if(seconds>=MAX_RECORDING_SECONDS)this.stop('2分に達したため録画を保存しました。続けて撮影できます。');},250);
  }
  async stop(message){if(this.recorder&&this.recorder.state!=='inactive'){this.change('stopping',{message});this.unschedule(this.timer);this.timer=null;this.recorder.stop();}if(this.recorder)await this.stopPromise;}
  async close(){++this.generation;this.opening=false;await this.stop();if(this.stream){this.stream.getTracks().forEach(track=>{track.onended=null;track.stop();});this.stream=null;}this.change('closed');}
}
