async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  const demoPort = Number(process.env.SPATIAL_DEMO_PORT || 18001);
  const sceneId = process.env.SPATIAL_ROOM_SCENE || "relocalization-smoke";
  const pageUrl = `http://127.0.0.1:${demoPort}/?scene=${encodeURIComponent(sceneId)}`;
  const target = await fetch(
    "http://127.0.0.1:9222/json/new?" + encodeURIComponent(pageUrl),
    { method: "PUT" }
  ).then(r => r.json());
  if (!target?.webSocketDebuggerUrl) throw new Error("Could not open fresh V2 E2E tab");

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
      events.push({
        text: details.text,
        description: details.exception?.description || null,
        url: details.url || null,
        line: details.lineNumber ?? null,
        column: details.columnNumber ?? null,
      });
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
  await sleep(1200);

  await evaluate(`location.hash = "room"; true`);
  for (let i = 0; i < 12; i++) {
    await sleep(150);
    if (await evaluate(`document.body.dataset.screen === "room"`)) break;
  }

  let targetEntity = null;
  for (let i = 0; i < 20 && !targetEntity; i++) {
    await sleep(250);
    targetEntity = await evaluate(`(async () => {
      const response = await fetch("/viewer/state/" + encodeURIComponent(
        new URLSearchParams(location.search).get("scene")
      ));
      const state = await response.json();
      const canvas = document.querySelector("#viewport canvas");
      const rect = canvas.getBoundingClientRect();
      for (const entity of state.entities) {
        const p = window.__spatialMemoryDebug.projectWorldToClient(entity.position);
        if (p.visible && p.x >= rect.left + 8 && p.x <= rect.right - 8 &&
            p.y >= rect.top + 8 && p.y <= rect.bottom - 8) {
          return { entity, point: p };
        }
      }
      return null;
    })()`);
  }
  if (!targetEntity) throw new Error("No visible entity available for 3D click test");
  const preMode = await evaluate(`window.__spatialMemoryDebug.selectionSnapshot().mode`);
  if (preMode) {
    await evaluate(`document.getElementById("select3dBtn").click(); true`);
    await sleep(100);
  }

  const before = await evaluate(`(() => ({
    screen: document.body.dataset.screen,
    buttonText: document.getElementById("select3dBtn").textContent,
    sheetHidden: document.getElementById("selectionSheet").hidden,
    mode: window.__spatialMemoryDebug.selectionSnapshot().mode
  }))()`);

  await evaluate(`document.getElementById("select3dBtn").click(); true`);
  await sleep(100);
  const point = targetEntity.point;
  await send("Input.dispatchMouseEvent", {
    type: "mousePressed",
    x: point.x,
    y: point.y,
    button: "left",
    buttons: 1,
    clickCount: 1,
  });
  await send("Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x: point.x,
    y: point.y,
    button: "left",
    buttons: 0,
    clickCount: 1,
  });
  await sleep(300);

  const selected = await evaluate(`(() => ({
    snapshot: window.__spatialMemoryDebug.selectionSnapshot(),
    sheetHidden: document.getElementById("selectionSheet").hidden,
    status: document.getElementById("selectionStatus").textContent,
    prefilledName: document.getElementById("selectionName").value,
    modeButtonText: document.getElementById("select3dBtn").textContent
  }))()`);

  if (!selected.snapshot.mode || !selected.snapshot.selected || !selected.snapshot.marker ||
      selected.sheetHidden || !["entity", "point-cloud"].includes(selected.snapshot.source)) {
    throw new Error("3D canvas tap did not produce a valid highlighted selection: " + JSON.stringify(selected));
  }
  const testName = "3D登録テスト";
  await evaluate(`(() => {
    const input = document.getElementById("selectionName");
    input.value = "${testName}";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    document.getElementById("save3dSelectionBtn").click();
    return true;
  })()`);

  let saved = null;
  for (let i = 0; i < 20; i++) {
    await sleep(300);
    saved = await evaluate(`(() => ({
      status: document.getElementById("selectionStatus").textContent,
      answer: document.getElementById("answer").textContent,
      entityText: document.getElementById("entities").textContent,
      snapshot: window.__spatialMemoryDebug.selectionSnapshot()
    }))()`);
    if (saved.status.includes("記憶しました")) break;
  }
  if (!saved?.status.includes("記憶しました: " + testName) ||
      !saved.entityText.includes(testName)) {
    throw new Error("3D direct label was not persisted: " + JSON.stringify(saved));
  }

  await evaluate(`document.getElementById("select3dBtn").click(); true`);
  await sleep(100);
  const ended = await evaluate(`(() => ({
    snapshot: window.__spatialMemoryDebug.selectionSnapshot(),
    buttonText: document.getElementById("select3dBtn").textContent,
    sheetHidden: document.getElementById("selectionSheet").hidden
  }))()`);
  if (ended.snapshot.mode || ended.snapshot.selected || !ended.sheetHidden) {
    throw new Error("3D selection mode did not cleanly exit");
  }

  if (events.length) {
    throw new Error("Unexpected browser exceptions: " + JSON.stringify(events));
  }
  console.log(JSON.stringify({ before, targetEntity, selected, saved, ended, events }, null, 2));
  ws.close();
  await fetch("http://127.0.0.1:9222/json/close/" + target.id);
}

main().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
