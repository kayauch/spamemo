async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  const demoPort = Number(process.env.SPATIAL_DEMO_PORT || 18001);
  const targets = await fetch("http://127.0.0.1:9222/json").then(r => r.json());
  const target = targets.find(t =>
    t.type === "page" && t.url.includes(`127.0.0.1:${demoPort}`)
  );
  if (!target) throw new Error("Spatial Memory V2 page not found");

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  let nextId = 1;
  const pending = new Map();
  const events = [];
  ws.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const job = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? job.reject(new Error(JSON.stringify(msg.error))) : job.resolve(msg.result);
      return;
    }
    if (msg.method === "Runtime.exceptionThrown") {
      events.push(msg.params.exceptionDetails.text);
    }
  };
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = reject;
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const result = await send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
      userGesture: true,
    });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
    return result.result.value;
  };

  await send("Runtime.enable");
  await send("Page.enable");
  await sleep(1500);

  await evaluate(`(() => {
    location.hash = "memory";
    return true;
  })()`);
  for (let i = 0; i < 12; i++) {
    await sleep(150);
    const screen = await evaluate(`document.body.dataset.screen`);
    if (screen === "memory") break;
  }

  const setup = await evaluate(`(() => {
    const slider = document.getElementById("captureFrame");
    slider.value = String(Math.min(10, Number(slider.max || 0)));
    slider.dispatchEvent(new Event("input", { bubbles: true }));
    document.getElementById("captureName").value = "ロボット";
    document.getElementById("capturePrompt").value = "";
    return {
      screen: document.body.dataset.screen,
      frame: slider.value,
      max: slider.max,
      fileInputsVisible: [...document.querySelectorAll('[data-screen-panel="memory"] input[type="file"]')]
        .some(element => getComputedStyle(element).display !== "none"),
      audioButtonsVisible: !!document.querySelector('[data-screen-panel="memory"] #recordBtn')
    };
  })()`);
  await sleep(900);

  if (setup.screen !== "memory") throw new Error("Memory screen did not open");
  if (setup.fileInputsVisible || setup.audioButtonsVisible) {
    throw new Error("Legacy upload/audio controls leaked into Memory screen");
  }
  await evaluate(`document.getElementById("detectFrameBtn").click(); true`);
  let detected = null;
  for (let i = 0; i < 24; i++) {
    await sleep(500);
    detected = await evaluate(`(() => ({
      status: document.getElementById("labelStatus").textContent,
      saveDisabled: document.getElementById("saveLabelBtn").disabled,
      overlayWidth: document.getElementById("frameOverlay").width,
      overlayHeight: document.getElementById("frameOverlay").height,
      previewWidth: document.getElementById("framePreview").naturalWidth,
      previewHeight: document.getElementById("framePreview").naturalHeight
    }))()`);
    if (!detected.saveDisabled || detected.status.includes("見つかりません")) break;
  }
  if (!detected || detected.saveDisabled) {
    throw new Error("Frame labeling candidate was not detected: " + JSON.stringify(detected));
  }
  if (detected.overlayWidth < 1 || detected.overlayHeight < 1 ||
      detected.previewWidth < 1 || detected.previewHeight < 1) {
    throw new Error("Frame preview/overlay did not render");
  }

  await evaluate(`document.getElementById("saveLabelBtn").click(); true`);
  let saved = null;
  for (let i = 0; i < 20; i++) {
    await sleep(400);
    saved = await evaluate(`(() => ({
      status: document.getElementById("labelStatus").textContent,
      answer: document.getElementById("answer").textContent,
      saveDisabled: document.getElementById("saveLabelBtn").disabled
    }))()`);
    if (saved.status.includes("記憶しました")) break;
  }
  if (!saved?.status.includes("記憶しました: ロボット")) {
    throw new Error("Text-only frame labeling did not save: " + JSON.stringify(saved));
  }
  console.log(JSON.stringify({ setup, detected, saved, events }, null, 2));
  ws.close();
}

main().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
