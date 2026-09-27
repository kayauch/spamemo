async function main() {
  const demoPort = Number(process.env.SPATIAL_DEMO_PORT || 18001);
  const targets = await fetch("http://127.0.0.1:9222/json").then(r => r.json());
  const target = targets.find(t => t.type === "page" && t.url.includes(`127.0.0.1:${demoPort}`));
  if (!target) throw new Error("Spatial Memory page not found");

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  let nextId = 1;
  const pending = new Map();
  const events = [];

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) reject(new Error(JSON.stringify(msg.error)));
      else resolve(msg.result);
      return;
    }
    if (msg.method === "Runtime.exceptionThrown") {
      events.push({ type: "exception", detail: msg.params.exceptionDetails.text });
    }
    if (msg.method === "Runtime.consoleAPICalled") {
      const values = (msg.params.args || []).map(a => a.value ?? a.description ?? "");
      events.push({ type: "console", level: msg.params.type, values });
    }
  };

  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });

  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });

  const evalExpr = async (expression, awaitPromise = true) => {
    const result = await send("Runtime.evaluate", {
      expression, awaitPromise, returnByValue: true, userGesture: true,
    });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text || "Runtime evaluation failed");
    return result.result.value;
  };

  await send("Runtime.enable");
  await send("Page.enable");
  await new Promise(r => setTimeout(r, 2000));

  const initial = await evalExpr(`(() => ({
    title: document.title,
    scene: document.getElementById("sceneLabel")?.textContent,
    entityCards: document.querySelectorAll("#entities .card").length,
    canvases: document.querySelectorAll("canvas").length,
    answer: document.getElementById("answer")?.textContent,
    cameraStatus: document.getElementById("cameraStatus")?.textContent,
    cameraButtonText: document.getElementById("cameraStartBtn")?.textContent,
    cameraButtonState: document.getElementById("cameraStartBtn")?.dataset.cameraState,
    cameraButtonDisabled: document.getElementById("cameraStartBtn")?.disabled
  }))()`);

  await evalExpr(`document.querySelector('[data-screen-target="find"]').click(); true`);
  await new Promise(r => setTimeout(r, 300));
  await evalExpr(`document.getElementById("cameraStartBtn").click(); true`);
  const cameraStarting = await evalExpr(`(() => ({
    buttonText: document.getElementById("cameraStartBtn").textContent,
    buttonState: document.getElementById("cameraStartBtn").dataset.cameraState,
    buttonDisabled: document.getElementById("cameraStartBtn").disabled
  }))()`);
  await new Promise(r => setTimeout(r, 4000));
  const camera = await evalExpr(`(() => ({
    status: document.getElementById("cameraStatus").textContent,
    videoWidth: document.getElementById("cameraVideo").videoWidth,
    videoHeight: document.getElementById("cameraVideo").videoHeight,
    readyState: document.getElementById("cameraVideo").readyState,
    buttonText: document.getElementById("cameraStartBtn").textContent,
    buttonState: document.getElementById("cameraStartBtn").dataset.cameraState,
    buttonDisabled: document.getElementById("cameraStartBtn").disabled,
    activeClass: document.getElementById("cameraStartBtn").classList.contains("camera-active"),
    liveTracks: window.__spatialMemoryDebug.resourceSnapshot().cameraTracks
  }))()`);

  await evalExpr(`document.getElementById("query").value="中央ターゲットどこ？"; document.getElementById("query").dispatchEvent(new Event("input",{bubbles:true})); document.getElementById("cameraLocateBtn").click(); true`);
  await new Promise(r => setTimeout(r, 9000));
  const located = await evalExpr(`(() => ({
    status: document.getElementById("cameraStatus").textContent,
    overlayWidth: document.getElementById("cameraOverlay").width,
    overlayHeight: document.getElementById("cameraOverlay").height,
    currentPosition: [
      document.getElementById("sx").value,
      document.getElementById("sy").value,
      document.getElementById("sz").value
    ]
  }))()`);

  await evalExpr(`document.getElementById("cameraTrackBtn").click(); true`);
  await new Promise(r => setTimeout(r, 3500));
  const tracking = await evalExpr(`(() => ({
    status: document.getElementById("cameraStatus").textContent,
    startDisabled: document.getElementById("cameraTrackBtn").disabled,
    stopDisabled: document.getElementById("cameraTrackStopBtn").disabled
  }))()`);
  await evalExpr(`document.getElementById("cameraTrackStopBtn").click(); true`);
  await evalExpr(`document.getElementById("cameraStartBtn").click(); true`);
  const cameraStopping = await evalExpr(`(() => ({
    buttonText: document.getElementById("cameraStartBtn").textContent,
    buttonState: document.getElementById("cameraStartBtn").dataset.cameraState,
    buttonDisabled: document.getElementById("cameraStartBtn").disabled
  }))()`);
  await new Promise(r => setTimeout(r, 500));
  const cameraStopped = await evalExpr(`(() => ({
    status: document.getElementById("cameraStatus").textContent,
    buttonText: document.getElementById("cameraStartBtn").textContent,
    buttonState: document.getElementById("cameraStartBtn").dataset.cameraState,
    buttonDisabled: document.getElementById("cameraStartBtn").disabled,
    activeClass: document.getElementById("cameraStartBtn").classList.contains("camera-active"),
    liveTracks: window.__spatialMemoryDebug.resourceSnapshot().cameraTracks,
    trackingStartDisabled: document.getElementById("cameraTrackBtn").disabled,
    trackingStopDisabled: document.getElementById("cameraTrackStopBtn").disabled
  }))()`);

  await evalExpr(`document.querySelector('[data-screen-target="memory"]').click(); true`);
  await new Promise(r => setTimeout(r, 200));
  await evalExpr(`document.getElementById("recordBtn").click(); true`);
  await new Promise(r => setTimeout(r, 1200));
  await evalExpr(`document.getElementById("stopBtn").click(); true`);
  await new Promise(r => setTimeout(r, 1000));
  const audio = await evalExpr(`(() => ({
    status: document.getElementById("captureStatus").textContent,
    recordDisabled: document.getElementById("recordBtn").disabled,
    stopDisabled: document.getElementById("stopBtn").disabled
  }))()`);

  const finalState = await evalExpr(`(() => ({
    answer: document.getElementById("answer").textContent,
    entityCards: document.querySelectorAll("#entities .card").length,
    pointCloudCanvasPresent: document.querySelector("#viewport canvas") !== null
  }))()`);

  if (initial.cameraButtonState !== "idle" || initial.cameraButtonText !== "カメラ開始" || initial.cameraButtonDisabled) {
    throw new Error("camera button initial state mismatch");
  }
  if (cameraStarting.buttonState !== "starting" || !cameraStarting.buttonDisabled ||
      !cameraStarting.buttonText.includes("起動中")) {
    throw new Error("camera starting state is not locked");
  }
  if (camera.buttonState !== "active" || camera.buttonText !== "カメラ停止" ||
      camera.buttonDisabled || !camera.activeClass || camera.liveTracks !== 1) {
    throw new Error("camera active UI does not match live MediaStream");
  }
  if (cameraStopping.buttonState !== "stopping" || !cameraStopping.buttonDisabled ||
      !cameraStopping.buttonText.includes("停止中")) {
    throw new Error("camera stopping state is not locked");
  }
  if (cameraStopped.buttonState !== "idle" || cameraStopped.buttonText !== "カメラ開始" ||
      cameraStopped.buttonDisabled || cameraStopped.activeClass || cameraStopped.liveTracks !== 0 ||
      cameraStopped.trackingStartDisabled || !cameraStopped.trackingStopDisabled) {
    throw new Error("camera stop UI/resources are not synchronized");
  }

  console.log(JSON.stringify({ target: { title: target.title, url: target.url }, initial, cameraStarting, camera, located, tracking, cameraStopping, cameraStopped, audio, finalState, events: events.slice(-20) }, null, 2));
  ws.close();
}

main().catch(err => { console.error(err.stack || err); process.exit(1); });