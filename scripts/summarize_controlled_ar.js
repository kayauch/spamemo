const fs = require("fs");
const path = require("path");
const root = path.join(__dirname, "..", "test-assets");
const rows = [];
for (let step = 1; step <= 6; step++) {
  const file = path.join(root, `ar-cont-${step}.json`);
  const j = JSON.parse(fs.readFileSync(file, "utf8"));
  rows.push({
    step,
    mode: j.mode,
    ref: j.pose?.reference_frame,
    rotation: j.pose?.image_rotation,
    inliers: j.pose?.inliers,
    matches: j.pose?.matches,
    reproj: j.pose ? Number(j.pose.reprojection_error.toFixed(3)) : null,
    confidence: j.pose ? Number(j.pose.confidence.toFixed(3)) : null,
    seconds: j.pose ? Number(j.pose.seconds.toFixed(3)) : null,
    nearest_camera_m: j.pose ? Number(j.pose.nearest_reference_camera_distance.toFixed(3)) : null,
    smooth: j.smoothing?.applied ?? null,
    on_screen: j.guidance?.on_screen ?? null,
    distance: j.guidance ? Number(j.guidance.distance.toFixed(3)) : null,
    norm_x: j.guidance ? Number(j.guidance.normalized_screen_target.x.toFixed(3)) : null,
    norm_y: j.guidance ? Number(j.guidance.normalized_screen_target.y.toFixed(3)) : null,
    det_score: j.detection ? Number(j.detection.score.toFixed(3)) : null,
    det_label: j.detection?.label ?? null,
    det_bbox: j.detection?.bbox_xyxy ?? null
  });
}
console.log(JSON.stringify(rows, null, 2));
