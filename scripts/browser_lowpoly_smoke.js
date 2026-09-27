async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  const demoPort = Number(process.env.SPATIAL_DEMO_PORT || 18001);
  const sceneId =
    process.env.SPATIAL_LOWPOLY_SCENE ||
    "lowpoly-smoke";
  const pageUrl =
    "http://127.0.0.1:" + demoPort +
    "/?scene=" + encodeURIComponent(sceneId) + "#room";

  const target = await fetch(
    "http://127.0.0.1:9222/json/new?" + encodeURIComponent(pageUrl),
    { method: "PUT" }
  ).then(r => r.json());
  if (!target?.webSocketDebuggerUrl) {
    throw new Error("Could not open low-poly smoke tab");
  }

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
      const details = msg.params.exceptionDetails;
      events.push(details.exception?.description || details.text);
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
  await send("Emulation.setDeviceMetricsOverride", {
    width: 390,
    height: 844,
    deviceScaleFactor: 3,
    mobile: true,
  });
  await send("Page.reload", { ignoreCache: true });

  let resources = null;
  for (let i = 0; i < 50; i++) {
    await sleep(250);
    resources = await evaluate(
      `window.__spatialMemoryDebug?.resourceSnapshot() || null`
    );
    if (resources?.spatialModelLoaded) break;
  }
  if (!resources?.spatialModelLoaded) {
    throw new Error("Low-poly spatial model did not load: " + JSON.stringify(resources));
  }
  if (resources.pointCloudLoaded || resources.pointCount !== 0) {
    throw new Error("Point cloud loaded despite low-poly model availability");
  }
  if (resources.spatialRenderMode !== "component-first") {
    throw new Error("Viewer did not enter component-first mode: " +
      JSON.stringify(resources));
  }
  if (!resources.roomFrameSnapEnabled ||
      !resources.gridRoomAligned ||
      Number(resources.gridRoomUpAlignment || 0) < 0.999) {
    throw new Error("Room-frame grid alignment is invalid: " +
      JSON.stringify(resources));
  }
  if (!resources.shellSupportLayer ||
      !(resources.shellOpacity > 0 && resources.shellOpacity <= 0.15)) {
    throw new Error("Poisson shell is not a low-opacity support layer");
  }
  if (resources.spatialComponentCount < 3) {
    throw new Error("Not enough selectable low-poly components");
  }
  if (resources.spatialComponents.some(item =>
    !["structural_regularized", "object_regularized"].includes(item.source)
  )) {
    throw new Error("Raw components leaked into component-first rendering");
  }
  if (resources.spatialShellTriangles < 1000 ||
      resources.spatialShellTriangles > 25000) {
    throw new Error("Unexpected low-poly triangle budget: " +
      resources.spatialShellTriangles);
  }

  const component =
    resources.spatialComponents.find(item => item.kind === "object_region") ||
    resources.spatialComponents.find(item => item.kind === "horizontal_surface") ||
    resources.spatialComponents[0];
  const projected = await evaluate(
    `window.__spatialMemoryDebug.projectWorldToClient(${JSON.stringify(component.center)})`
  );
  if (!projected?.visible) {
    throw new Error("Chosen low-poly component is outside the camera view");
  }

  await evaluate(`document.getElementById("select3dBtn").click(); true`);
  await send("Input.dispatchMouseEvent", {
    type: "mousePressed",
    x: projected.x,
    y: projected.y,
    button: "left",
    buttons: 1,
    clickCount: 1,
  });
  await send("Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x: projected.x,
    y: projected.y,
    button: "left",
    buttons: 0,
    clickCount: 1,
  });
  await sleep(250);

  const selected = await evaluate(`(() => ({
    selection: window.__spatialMemoryDebug.selectionSnapshot(),
    status: document.getElementById("selectionStatus").textContent,
    sheetHidden: document.getElementById("selectionSheet").hidden
  }))()`);
  if (selected.selection.source !== "component" ||
      !selected.selection.componentId ||
      selected.sheetHidden) {
    throw new Error("Low-poly component was not selected: " + JSON.stringify(selected));
  }

  const testName = "低ポリ登録テスト";
  await evaluate(`(() => {
    const input = document.getElementById("selectionName");
    input.value = "${testName}";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    document.getElementById("save3dSelectionBtn").click();
    return true;
  })()`);

  let saved = null;
  for (let i = 0; i < 30; i++) {
    await sleep(250);
    saved = await evaluate(`(() => ({
      status: document.getElementById("selectionStatus").textContent,
      entityText: document.getElementById("entities").textContent
    }))()`);
    if (saved.status.includes("記憶しました")) break;
  }
  if (!saved?.status.includes("記憶しました: " + testName)) {
    throw new Error("Low-poly component label was not persisted: " +
      JSON.stringify(saved));
  }

  const viewerState = await fetch(
    "http://127.0.0.1:" + demoPort +
    "/viewer/state/" + encodeURIComponent(sceneId)
  ).then(r => r.json());
  const entity = viewerState.entities.find(item => item.name === testName);
  if (!entity) throw new Error("Saved low-poly Entity not found");
  if (entity.metadata?.spatial_component_id !== selected.selection.componentId) {
    throw new Error("Entity lost spatial_component_id metadata: " +
      JSON.stringify(entity.metadata));
  }

  if (events.length) {
    throw new Error("Unexpected browser exceptions: " + JSON.stringify(events));
  }

  console.log(JSON.stringify({
    resources: {
      spatialModelLoaded: resources.spatialModelLoaded,
      pointCloudLoaded: resources.pointCloudLoaded,
      renderMode: resources.spatialRenderMode,
      roomFrameSnapEnabled: resources.roomFrameSnapEnabled,
      roomFrameConfidence: resources.roomFrameConfidence,
      gridRoomAligned: resources.gridRoomAligned,
      gridRoomUpAlignment: resources.gridRoomUpAlignment,
      shellSupportLayer: resources.shellSupportLayer,
      shellOpacity: resources.shellOpacity,
      shellVertices: resources.spatialShellVertices,
      shellTriangles: resources.spatialShellTriangles,
      componentCount: resources.spatialComponentCount,
      gridSize: resources.gridSize,
      cameraDistanceToScene: resources.cameraDistanceToScene,
    },
    chosenComponent: component,
    selected,
    saved,
    entity: {
      id: entity.id,
      name: entity.name,
      position: entity.position,
      spatial_component_id: entity.metadata?.spatial_component_id,
      spatial_component_kind: entity.metadata?.spatial_component_kind,
    },
    events,
  }, null, 2));

  ws.close();
  await fetch("http://127.0.0.1:9222/json/close/" + target.id);
}

main().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
