// Check both the visible orientation and the actual camera frame. Rotating
// the CSS preview alone must never make a portrait recording look acceptable.
export function captureOrientation(sample, baseline = null) {
  if (!sample.landscape) return {allowed: false, reason: 'portrait', message: 'スマホを横にしてください。横向きになると録画できます。'};
  if (!(sample.width > 0 && sample.height > 0)) return {allowed: false, reason: 'loading', message: 'カメラの映像を確認しています。'};
  if (sample.width <= sample.height) return {allowed: false, reason: 'portrait-frame', message: 'カメラの映像が縦向きです。一度撮影を終了し、横向きのままカメラを開き直してください。'};
  if (baseline && ((Number.isFinite(baseline.angle) && Number.isFinite(sample.angle) && baseline.angle !== sample.angle) || baseline.width !== sample.width || baseline.height !== sample.height)) {
    return {allowed: false, reason: 'changed', message: 'カメラの向きや映像のサイズが変わりました。向きを固定して、次の動画を撮影してください。'};
  }
  return {allowed: true, reason: 'landscape', message: ''};
}
