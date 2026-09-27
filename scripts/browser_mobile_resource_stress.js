async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  const sceneId = process.argv[2] || "749ad0af-a114-4efc-b2cb-e95fc1c258fc";
  const targets = await fetch("http://127.0.0.1:9222/json").then(r => r.json());
  const target = targets.find(t =>
    t.type === "page" && t.url.includes("127.0.0.1:18001") && t.url.includes(sceneId)
  );
  if (!target) throw new Error("V2 stress-test page not found");

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
  const evaluate = async expression => {
    const result = await send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.text || "Runtime evaluation failed");
    }
    return result.result.value;
  };

  await send("Runtime.enable");
  await send("Page.enable");
  await send("Performance.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: 390,
    height: 844,
    deviceScaleFactor: 3,
    mobile: true,
  });

  async function waitForViewerAsset(timeoutMs = 15000) {
    const started = Date.now();
    while (Date.now() - started < timeoutMs) {
      const state = await evaluate(`(() => {
        const debug = window.__spatialMemoryDebug;
        return debug ? debug.resourceSnapshot() : null;
      })()`);
      if (
        state?.spatialModelLoaded &&
        state.spatialShellTriangles > 0
      ) {
        return state;
      }
      if (state?.pointCloudLoaded && state.pointCount > 0) return state;
      await sleep(250);
    }
    throw new Error("viewer asset load timeout");
  }

  async function metrics() {
    const result = await send("Performance.getMetrics");
    const map = Object.fromEntries(result.metrics.map(item => [item.name, item.value]));
    const dom = await send("Memory.getDOMCounters");
    return {
      jsHeapUsed: map.JSHeapUsedSize || null,
      jsHeapTotal: map.JSHeapTotalSize || null,
      nodes: dom.nodes,
      documents: dom.documents,
      listeners: dom.jsEventListeners,
    };
  }

  const rounds = [];
  for (let round = 1; round <= 3; round++) {
    await send("Page.reload", { ignoreCache: false });
    await sleep(1200);
    const loaded = await waitForViewerAsset();

    await evaluate(`document.getElementById("entities").scrollIntoView({block:"end"}); true`);
    await sleep(300);
    await evaluate(`document.getElementById("viewport").scrollIntoView({block:"start"}); true`);
    await sleep(300);

    const before = {
      resources: await evaluate("window.__spatialMemoryDebug.resourceSnapshot()"),
      metrics: await metrics(),
    };

    await evaluate("window.__spatialMemoryDebug.suspendForBackground(); true");
    await sleep(250);
    const suspended = await evaluate("window.__spatialMemoryDebug.resourceSnapshot()");

    await evaluate("window.__spatialMemoryDebug.resumeAfterBackground(); true");
    const resumedResources = await waitForViewerAsset();
    const after = {
      resources: resumedResources,
      metrics: await metrics(),
    };
    rounds.push({ round, loaded, before, suspended, after });
  }
  const pointCounts = rounds.map(item => item.after.resources.pointCount);
  const shellTriangles = rounds.map(
    item => item.after.resources.spatialShellTriangles || 0
  );
  const componentCounts = rounds.map(
    item => item.after.resources.spatialComponentCount || 0
  );
  const assetModes = rounds.map(item =>
    item.after.resources.spatialModelLoaded ? "lowpoly" : "point-cloud"
  );
  const renderModes = rounds.map(
    item => item.after.resources.spatialRenderMode || "point-cloud"
  );
  const geometryCounts = rounds.map(item => item.after.resources.geometries);
  const heapValues = rounds.map(item => item.after.metrics.jsHeapUsed).filter(Number.isFinite);

  for (const item of rounds) {
    const resources = item.after.resources;
    if (resources.spatialModelLoaded) {
      if (
        resources.pointCloudLoaded ||
        resources.pointCount !== 0 ||
        resources.spatialShellTriangles < 1000 ||
        resources.spatialShellTriangles > 25000 ||
        resources.spatialComponentCount < 1 ||
        resources.spatialComponentCount > 40
      ) {
        throw new Error("mobile low-poly budget/state invalid");
      }
      if (resources.roomFrameSnapEnabled) {
        if (
          resources.spatialRenderMode !== "component-first" ||
          !resources.gridRoomAligned ||
          Number(resources.gridRoomUpAlignment || 0) < 0.999 ||
          !resources.shellSupportLayer ||
          !(resources.shellOpacity > 0 && resources.shellOpacity <= 0.15)
        ) {
          throw new Error("room-frame component-first viewer state invalid");
        }
      }
    } else if (resources.pointCount < 1 || resources.pointCount > 100000) {
      throw new Error("mobile point-cloud budget exceeded");
    }
  }
  if (rounds.some(item =>
    item.suspended.pointCount !== 0 ||
    item.suspended.pointCloudLoaded ||
    item.suspended.spatialModelLoaded
  )) {
    throw new Error("viewer geometry was not released while suspended");
  }
  if (Math.max(...geometryCounts) - Math.min(...geometryCounts) > 3) {
    throw new Error("WebGL geometry count grows across cycles");
  }
  if (heapValues.length === 3 && heapValues[2] > heapValues[0] * 2 + 20_000_000) {
    throw new Error("JS heap grew excessively across cycles");
  }

  console.log(JSON.stringify({
    sceneId,
    rounds,
    summary: {
      assetModes,
      renderModes,
      pointCounts,
      shellTriangles,
      componentCounts,
      geometryCounts,
      heapDeltaBytes: heapValues.length === 3 ? heapValues[2] - heapValues[0] : null,
    },
  }, null, 2));
  ws.close();
}

main().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
