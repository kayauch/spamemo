const fs = require("fs");
const path = require("path");
const { spawn, spawnSync } = require("child_process");

async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function main() {
  const project = path.resolve(__dirname, "..");
  const stateText = fs.readFileSync(
    path.join(project, "test-assets", "mobile-demo-state.json"), "utf8"
  ).replace(/^\uFEFF/, "");
  const state = JSON.parse(stateText);
  const candidates = [
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  ];
  const edge = candidates.find(fs.existsSync);
  if (!edge) throw new Error("Microsoft Edge not found");

  const fakeCamera = path.join(project, "test-assets", "fake-camera.y4m");
  const profile = path.join(project, "test-assets", "public-mobile-edge-profile");
  fs.mkdirSync(profile, { recursive: true });
  const url = `${state.base_url}/?scene=${encodeURIComponent(state.scene)}&access_token=${state.token}`;
  const child = spawn(edge, [
    "--headless=new",
    "--remote-debugging-port=9223",
    `--user-data-dir=${profile}`,
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    `--use-file-for-fake-video-capture=${fakeCamera}`,
    "--autoplay-policy=no-user-gesture-required",
    url,
  ], { stdio: "ignore", windowsHide: true });

  try {
    let targets = null;
    for (let i = 0; i < 30; i++) {
      await sleep(500);
      try {
        targets = await fetch("http://127.0.0.1:9223/json").then(r => r.json());
        if (targets.some(t => t.type === "page" && t.url.includes("trycloudflare.com"))) break;
      } catch {}
    }
    if (!targets) throw new Error("Public mobile Edge CDP did not start");
    const target = targets.find(t => t.type === "page" && t.url.includes("trycloudflare.com"));
    if (!target) throw new Error("Public V2 target not found");

    const ws = new WebSocket(target.webSocketDebuggerUrl);
    let nextId = 1;
    const pending = new Map();
    ws.onmessage = event => {
      const msg = JSON.parse(event.data);
      if (!msg.id || !pending.has(msg.id)) return;
      const job = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? job.reject(new Error(JSON.stringify(msg.error))) : job.resolve(msg.result);
    };
    await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
    const send = (method, params = {}) => new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    });
    const evaluate = async expression => {
      const result = await send("Runtime.evaluate", {
        expression, awaitPromise: true, returnByValue: true, userGesture: true,
      });
      if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
      return result.result.value;
    };

    await send("Runtime.enable");
    await send("Page.enable");
    await send("Emulation.setDeviceMetricsOverride", {
      width: 390, height: 844, deviceScaleFactor: 3, mobile: true,
    });
    await send("Page.reload", { ignoreCache: true });
    await sleep(6000);

    const initial = await evaluate(`(() => ({
      title: document.title,
      secureContext: isSecureContext,
      mediaDevices: !!navigator.mediaDevices?.getUserMedia,
      mobileMedia: matchMedia("(max-width: 820px)").matches,
      scene: document.getElementById("sceneLabel")?.textContent,
      navDisplay: getComputedStyle(document.querySelector(".mobile-nav")).display,
      urlHasAccessToken: location.search.includes("access_token=")
    }))()`);

    await evaluate(`document.querySelector('[data-screen-target="find"]').click(); true`);
    await sleep(300);
    await evaluate(`document.getElementById("cameraStartBtn").click(); true`);
    await sleep(7000);
    const camera = await evaluate(`(() => ({
      status: document.getElementById("cameraStatus").textContent,
      width: document.getElementById("cameraVideo").videoWidth,
      height: document.getElementById("cameraVideo").videoHeight,
      readyState: document.getElementById("cameraVideo").readyState
    }))()`);

    await evaluate(`(() => {
      document.getElementById("query").value = "中央ターゲットどこ？";
      document.getElementById("query").dispatchEvent(new Event("input", { bubbles: true }));
      document.getElementById("cameraLocateBtn").click();
      return true;
    })()`);
    await sleep(11000);
    const located = await evaluate(`(() => ({
      status: document.getElementById("cameraStatus").textContent,
      x: document.getElementById("sx").value,
      y: document.getElementById("sy").value,
      z: document.getElementById("sz").value
    }))()`);

    console.log(JSON.stringify({ initial, camera, located }, null, 2));
    if (!initial.secureContext || !initial.mediaDevices || !initial.mobileMedia) {
      throw new Error("Public HTTPS mobile prerequisites failed");
    }
    if (initial.urlHasAccessToken) {
      throw new Error("Access token remained visible in the address bar");
    }
    if (camera.readyState < 2 || camera.width < 1) {
      throw new Error("Camera did not start over public HTTPS");
    }
    if (!located.status.includes("中央ターゲット")) {
      throw new Error("AR relocalization did not complete over public HTTPS");
    }
    ws.close();
  } finally {
    spawnSync("taskkill", ["/PID", String(child.pid), "/T", "/F"], { stdio: "ignore" });
  }
}

main().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
