async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  const demoPort = Number(process.env.SPATIAL_DEMO_PORT || 18001);
  const sceneId = process.env.SPATIAL_ROOM_SCENE || "relocalization-smoke";
  const base = "http://127.0.0.1:" + demoPort;
  const target = await fetch(
    "http://127.0.0.1:9222/json/new?" + encodeURIComponent(base + "/"),
    { method: "PUT" }
  ).then(r => r.json());

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  let nextId = 1;
  const pending = new Map();
  const requested = [];
  ws.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const job = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? job.reject(new Error(JSON.stringify(msg.error))) : job.resolve(msg.result);
      return;
    }
    if (msg.method === "Network.requestWillBeSent") requested.push(msg.params.request.url);
  };
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const result = await send("Runtime.evaluate", {
      expression, awaitPromise: true, returnByValue: true,
    });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
    return result.result.value;
  };

  await send("Runtime.enable");
  await send("Network.enable");
  await send("Network.setCacheDisabled", { cacheDisabled: true });
  await send("Page.enable");
  await send("Page.reload", { ignoreCache: true });
  await sleep(1200);

  for (let attempt = 0; attempt < 16; attempt++) {
    const ready = await evaluate(`(() =>
      [...document.querySelectorAll("#demoHealth .health-chip")]
        .some(el => el.textContent?.startsWith("API: ready"))
    )()`);
    if (ready) break;
    await sleep(250);
  }

  const home = await evaluate(`(() => ({
    title: document.title,
    heading: document.querySelector("h1")?.textContent,
    rooms: document.querySelectorAll(".room").length,
    createText: document.getElementById("createRoom")?.textContent,
    canvases: document.querySelectorAll("canvas").length,
    health: [...document.querySelectorAll("#demoHealth .health-chip")].map(
      el => ({ text: el.textContent, className: el.className })
    )
  }))()`);
  const homeRequests = [...requested];
  if (!home.heading?.includes("どの部屋")) throw new Error("home screen did not render");
  if (home.rooms < 1) throw new Error("saved rooms were not listed");
  if (home.canvases !== 0) throw new Error("home unexpectedly created a 3D canvas");
  if (!home.health.some(item => item.text?.startsWith("API: ready"))) {
    throw new Error("demo health API chip missing");
  }
  if (!home.health.some(item => item.text?.startsWith("GPU:"))) {
    throw new Error("demo health GPU chip missing");
  }
  if (!home.health.some(item => item.text?.startsWith("LLM:"))) {
    throw new Error("demo health LLM chip missing");
  }
  if (homeRequests.some(url => url.includes("/point-cloud"))) {
    throw new Error("home eagerly downloaded a point cloud");
  }

  requested.length = 0;
  await send("Page.navigate", { url: base + "/room/new" });
  await sleep(1200);
  const create = await evaluate(`(() => {
    const plan = window.__createRoomSampling?.deriveSamplingPlan(30, 0.08, 0.05);
    return {
      heading: document.querySelector("h1")?.textContent,
      fpsLabel: document.querySelector('label[for="sampleFps"]')?.textContent,
      frameLabel: document.querySelector('label[for="maxFrames"]')?.textContent,
      autoPlanStatus: document.getElementById("samplingPlan")?.textContent,
      manualChecked: document.getElementById("manualSampling")?.checked,
      fpsDisabled: document.getElementById("sampleFps")?.disabled,
      framesDisabled: document.getElementById("maxFrames")?.disabled,
      derivedPlan: plan,
      canvases: document.querySelectorAll("canvas").length
    };
  })()`);
  if (!create.heading?.includes("新しい部屋")) throw new Error("create-room screen missing");
  if (!create.fpsLabel || !create.frameLabel) throw new Error("sampling labels missing");
  if (create.manualChecked || !create.fpsDisabled || !create.framesDisabled) {
    throw new Error("Auto sampling is not the default");
  }
  if (!create.derivedPlan || create.derivedPlan.sampleFps <= 0 || create.derivedPlan.estimatedFrames < 2) {
    throw new Error("Auto sampling planner unavailable");
  }
  const manual = await evaluate(`(() => {
    const toggle = document.getElementById("manualSampling");
    toggle.checked = true;
    toggle.dispatchEvent(new Event("change", { bubbles: true }));
    return {
      fpsDisabled: document.getElementById("sampleFps").disabled,
      framesDisabled: document.getElementById("maxFrames").disabled
    };
  })()`);
  if (manual.fpsDisabled || manual.framesDisabled) {
    throw new Error("Advanced manual override did not enable numeric controls");
  }
  if (requested.some(url => url.includes("/point-cloud"))) {
    throw new Error("create-room eagerly downloaded a point cloud");
  }
  requested.length = 0;
  await send("Page.navigate", { url: base + "/?scene=" + encodeURIComponent(sceneId) });
  await sleep(3500);
  const workspace = await evaluate(`(() => ({
    title: document.title,
    scene: document.getElementById("sceneLabel")?.textContent,
    canvases: document.querySelectorAll("canvas").length
  }))()`);
  const pointCloudRequested = requested.some(url => url.includes("/point-cloud"));
  const spatialModelRequested = requested.some(url => url.includes("/spatial-model"));
  if (!workspace.scene?.includes(sceneId)) throw new Error("room workspace did not open");
  if (workspace.canvases < 1) throw new Error("workspace 3D canvas missing");
  if (!pointCloudRequested && !spatialModelRequested) {
    throw new Error("workspace did not request any 3D scene asset");
  }

  console.log(JSON.stringify({
    home,
    create,
    workspace,
    homeRequestedPointCloud: homeRequests.some(url => url.includes("/point-cloud")),
    workspaceRequestedPointCloud: pointCloudRequested,
    workspaceRequestedSpatialModel: spatialModelRequested,
  }, null, 2));
  ws.close();
  await fetch("http://127.0.0.1:9222/json/close/" + target.id);
}

main().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
