const params = new URLSearchParams(location.search);
if (params.has("access_token")) {
  params.delete("access_token");
  history.replaceState({}, "", location.pathname + (params.toString() ? "?" + params : ""));
}

const videoInput = document.getElementById("sceneVideo");
const button = document.getElementById("reconstructBtn");
const status = document.getElementById("reconstructStatus");
const planStatus = document.getElementById("samplingPlan");
const manualSampling = document.getElementById("manualSampling");
const sampleFpsInput = document.getElementById("sampleFps");
const maxFramesInput = document.getElementById("maxFrames");
let currentPlan = null;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

async function safeJson(response, fallbackMsg = "処理に失敗しました") {
  const text = await response.text();
  let json = null;
  try {
    json = JSON.parse(text);
  } catch {}
  if (!response.ok) {
    const detail = json?.detail || text || fallbackMsg;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (json === null) {
    throw new Error("サーバーから不正な応答を受信しました: " + text.slice(0, 100));
  }
  return json;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
function deriveSamplingPlan(durationSeconds, motionScore, sharpnessScore) {
  let sampleFps = 2.0;
  if (motionScore < 0.025) sampleFps = 1.5;
  else if (motionScore < 0.060) sampleFps = 2.0;
  else if (motionScore < 0.120) sampleFps = 2.5;
  else if (motionScore < 0.200) sampleFps = 3.0;
  else sampleFps = 3.5;

  const expected = Math.max(2, Math.ceil(durationSeconds * sampleFps));
  const maxFrames = clamp(expected, 20, 72);
  const estimatedFrames = Math.min(expected, maxFrames);
  const motionLabel = motionScore < 0.035 ? "ゆっくり" :
    motionScore < 0.110 ? "標準" :
    motionScore < 0.200 ? "速め" : "かなり速い";
  const sharpnessLabel = sharpnessScore < 0.018 ? "ブレ多め" :
    sharpnessScore < 0.040 ? "ややブレ" : "良好";

  return {
    durationSeconds,
    motionScore,
    sharpnessScore,
    sampleFps,
    maxFrames,
    estimatedFrames,
    motionLabel,
    sharpnessLabel,
  };
}

window.__createRoomSampling = { deriveSamplingPlan };
function waitFor(target, eventName) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(eventName + " timeout")), 8000);
    target.addEventListener(eventName, () => {
      clearTimeout(timer);
      resolve();
    }, { once: true });
    target.addEventListener("error", () => {
      clearTimeout(timer);
      reject(new Error("動画を読み込めませんでした"));
    }, { once: true });
  });
}

async function seekVideo(video, time) {
  if (Math.abs(video.currentTime - time) < 0.01) return;
  const promise = waitFor(video, "seeked");
  video.currentTime = time;
  await promise;
}

function frameFeatures(imageData, previousGray) {
  const { data, width, height } = imageData;
  const gray = new Uint8Array(width * height);
  let gradient = 0;
  let motion = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const p = i * 4;
      const value = Math.round(0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]);
      gray[i] = value;
      if (x > 0) gradient += Math.abs(value - gray[i - 1]);
      if (y > 0) gradient += Math.abs(value - gray[i - width]);
      if (previousGray) motion += Math.abs(value - previousGray[i]);
    }
  }
  const pixels = width * height;
  return {
    gray,
    sharpness: gradient / Math.max(1, pixels * 2 * 255),
    motion: previousGray ? motion / Math.max(1, pixels * 255) : null,
  };
}
async function analyzeVideo(file) {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.preload = "metadata";
  video.muted = true;
  video.playsInline = true;
  video.src = url;
  try {
    if (video.readyState < 1) await waitFor(video, "loadedmetadata");
    const duration = Number(video.duration);
    if (!Number.isFinite(duration) || duration <= 0) {
      throw new Error("動画の長さを取得できません");
    }

    const canvas = document.createElement("canvas");
    canvas.width = 160;
    canvas.height = 90;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const sampleCount = clamp(Math.ceil(duration / 2), 6, 12);
    const motions = [];
    const sharpness = [];
    let previousGray = null;

    for (let i = 0; i < sampleCount; i++) {
      const t = Math.min(
        Math.max(0, duration - 0.08),
        duration * ((i + 0.5) / sampleCount)
      );
      await seekVideo(video, t);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const features = frameFeatures(
        ctx.getImageData(0, 0, canvas.width, canvas.height),
        previousGray
      );
      previousGray = features.gray;
      sharpness.push(features.sharpness);
      if (features.motion !== null) motions.push(features.motion);
    }

    return deriveSamplingPlan(duration, median(motions), median(sharpness));
  } finally {
    URL.revokeObjectURL(url);
    video.removeAttribute("src");
    video.load();
  }
}
function applyPlan(plan) {
  currentPlan = plan;
  if (!manualSampling.checked) {
    sampleFpsInput.value = String(plan.sampleFps);
    maxFramesInput.value = String(plan.maxFrames);
  }
  planStatus.innerHTML =
    '<span class="ok">Auto設定</span><br>' +
    "動画: " + plan.durationSeconds.toFixed(1) + "秒 / " +
    "動き: " + plan.motionLabel + " / " +
    "画質: " + plan.sharpnessLabel + "<br>" +
    "抽出: " + plan.sampleFps.toFixed(1) + " FPS / 最大 " +
    plan.maxFrames + " frames / 使用予想 約" + plan.estimatedFrames + "枚";
}

function syncManualMode() {
  const manual = manualSampling.checked;
  sampleFpsInput.disabled = !manual;
  maxFramesInput.disabled = !manual;
  if (!manual && currentPlan) applyPlan(currentPlan);
}
manualSampling.addEventListener("change", syncManualMode);
syncManualMode();

videoInput.addEventListener("change", async () => {
  const file = videoInput.files[0];
  currentPlan = null;
  if (!file) {
    status.textContent = "動画を選択してください";
    planStatus.textContent = "動画を選ぶと、長さ・動き・ブレから自動設定を計算します。";
    return;
  }
  status.textContent = "選択済み: " + file.name;
  planStatus.textContent = "動画を軽量解析してAuto設定を計算中…";
  try {
    applyPlan(await analyzeVideo(file));
  } catch (error) {
    currentPlan = deriveSamplingPlan(30, 0.06, 0.04);
    applyPlan(currentPlan);
    planStatus.innerHTML += '<br><span class="warn">解析できなかったため安全な既定値を使用: ' +
      error.message + "</span>";
  }
});
async function reconstructRoom() {
  const video = videoInput.files[0];
  if (!video) throw new Error("部屋を撮影した動画を選択してください");
  if (!manualSampling.checked && !currentPlan) {
    planStatus.textContent = "Auto設定を計算中…";
    applyPlan(await analyzeVideo(video));
  }

  const form = new FormData();
  form.append("video", video);
  form.append("name", document.getElementById("sceneName").value.trim() || "New Room");
  form.append("sample_fps", sampleFpsInput.value || "2");
  form.append("max_frames", maxFramesInput.value || "60");
  form.append("point_stride", "4");
  form.append("quality_filter", "true");

  button.disabled = true;
  status.textContent = "VMで3Dシーンを生成中です…";
  try {
    const response = await fetch("/scenes/reconstruct", { method: "POST", body: form });
    const data = await safeJson(response, "3Dシーン生成に失敗しました");
    status.innerHTML =
      '<span class="ok">3D生成完了</span><br>' +
      (data.scene.reconstruction_model || data.metrics.model || "model unknown") +
      " ・ " + data.scene.frame_count + " frames<br>" +
      "動画音声からSpatial Memoryを作成中…";

    let audioMemory = null;
    try {
      const audioForm = new FormData();
      audioForm.append("language", "ja");
      audioForm.append("detection_threshold", "0.25");
      audioForm.append("text_threshold", "0.25");
      audioForm.append("auto_confirm_threshold", "0.60");
      const audioResponse = await fetch(
        "/scenes/" + encodeURIComponent(data.scene.id) + "/audio-memory",
        { method: "POST", body: audioForm }
      );
      audioMemory = await safeJson(audioResponse, "動画音声の記憶化に失敗しました");
      status.innerHTML =
        '<span class="ok">部屋の準備完了</span><br>' +
        data.scene.frame_count + " frames / " +
        "音声から自動記憶 " + audioMemory.confirmed_count + "件 / " +
        "確認待ち " + audioMemory.pending_count + "件";
    } catch (audioError) {
      console.warn("audio-memory skipped", audioError);
      status.innerHTML =
        '<span class="ok">3D生成完了</span><br>' +
        '<span class="warn">動画音声の記憶化はスキップ: ' + audioError.message + "</span>";
    }
    sessionStorage.setItem(
      "spatial:last-audio-memory:" + data.scene.id,
      JSON.stringify(audioMemory || {})
    );
    location.href = "/?scene=" + encodeURIComponent(data.scene.id);
  } finally {
    button.disabled = false;
  }
}

button.onclick = () => reconstructRoom().catch(error => {
  status.innerHTML = '<span class="warn">' + error.message + "</span>";
});
