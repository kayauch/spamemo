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
  ws.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (!msg.id || !pending.has(msg.id)) return;
    const job = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? job.reject(new Error(JSON.stringify(msg.error))) : job.resolve(msg.result);
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

  await send("Emulation.setDeviceMetricsOverride", {
    width: 390, height: 844, deviceScaleFactor: 3, mobile: true,
  });
  await send("Page.enable");
  await send("Page.reload", { ignoreCache: true });
  await new Promise(resolve => setTimeout(resolve, 5000));

  const evaluate = async expression => {
    const result = await send("Runtime.evaluate", {
      expression, returnByValue: true, userGesture: true,
    });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
    return result.result.value;
  };

  const initial = await evaluate(`(() => ({
    title: document.title,
    mobileMedia: matchMedia("(max-width: 820px)").matches,
    appDisplay: getComputedStyle(document.getElementById("app")).display,
    navDisplay: getComputedStyle(document.querySelector(".workspace-nav")).display,
    activeScreen: document.body.dataset.screen,
    activePanels: document.querySelectorAll(".screen.active").length,
    inputFontSize: getComputedStyle(document.getElementById("query")).fontSize,
    debugVisible: getComputedStyle(document.getElementById("debugNavBtn")).display !== "none",
    viewer: window.__spatialMemoryDebug?.resourceSnapshot() || null
  }))()`);

  await evaluate(`document.querySelector('[data-screen-target="find"]').click(); true`);
  await new Promise(resolve => setTimeout(resolve, 300));
  const find = await evaluate(`(() => ({
    activeScreen: document.body.dataset.screen,
    cameraWidth: document.getElementById("cameraWrap").getBoundingClientRect().width,
    viewportDisplay: getComputedStyle(document.getElementById("viewport")).display
  }))()`);

  await evaluate(`document.querySelector('[data-screen-target="memory"]').click(); true`);
  await new Promise(resolve => setTimeout(resolve, 300));
  const memory = await evaluate(`(() => ({
    activeScreen: document.body.dataset.screen,
    activePanels: document.querySelectorAll(".screen.active").length,
    viewportDisplay: getComputedStyle(document.getElementById("viewport")).display
  }))()`);

  if (!initial.mobileMedia || initial.navDisplay === "none" || initial.activePanels !== 1) {
    throw new Error("Mobile scene navigation did not activate");
  }
  if (parseFloat(initial.inputFontSize) < 16 || initial.debugVisible) {
    throw new Error("Mobile controls are not user-facing safe");
  }
  if (!initial.viewer?.pointCloudLoaded || !initial.viewer.viewerBounds) {
    throw new Error("Mobile viewer bounds were not prepared");
  }
  const size = initial.viewer.viewerBounds.size;
  const maxSpan = Math.max(size.x, size.y, size.z);
  const diagonal = Math.hypot(size.x, size.y, size.z);
  if (!(initial.viewer.gridSize >= maxSpan * 0.8 &&
        initial.viewer.gridSize <= maxSpan * 2.2)) {
    throw new Error("Scene grid is not normalized to scene bounds");
  }
  if (!(initial.viewer.cameraDistanceToScene >= diagonal * 0.55 &&
        initial.viewer.cameraDistanceToScene <= diagonal * 1.8)) {
    throw new Error("Camera fit is not normalized to scene bounds");
  }
  if (find.activeScreen !== "find" || find.cameraWidth < 1) {
    throw new Error("Find screen did not activate");
  }
  if (memory.activeScreen !== "memory" || memory.activePanels !== 1 || memory.viewportDisplay !== "none") {
    throw new Error("Memory screen did not isolate heavy viewer");
  }
  console.log(JSON.stringify({ initial, find, memory }, null, 2));
  ws.close();
}
main().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
