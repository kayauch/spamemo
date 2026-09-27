export const MAX_RECORDING_BYTES=90*1024*1024;
export const MAX_RECORDING_SECONDS=120;
export function recordingMime(Recorder){if(!Recorder)return null;return ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/mp4','video/webm'].find(type=>Recorder.isTypeSupported?.(type))||'';}
export function recordingExtension(mime){return String(mime).toLowerCase().includes('mp4')?'mp4':'webm';}
export function elapsedLabel(seconds){const n=Math.max(0,Math.floor(seconds));return `${Math.floor(n/60).toString().padStart(2,'0')}:${(n%60).toString().padStart(2,'0')}`;}
export function supportedZooms(capability){if(!capability||!Number.isFinite(capability.min)||!Number.isFinite(capability.max)||capability.max<=capability.min)return [];return [.5,1,2,3].filter(value=>value>=capability.min&&value<=capability.max&&(!capability.step||Math.abs((value-capability.min)/capability.step-Math.round((value-capability.min)/capability.step))<.001));}
export function recordingFile(blob,stamp=Date.now()){if(!blob?.size)throw Error('録画データを保存できませんでした。もう一度撮影してください。');if(blob.size>256*1024*1024)throw Error('録画が大きすぎます。短く分けて撮影してください。');return new File([blob],`room-${stamp}.${recordingExtension(blob.type)}`,{type:blob.type||'video/webm'});}
