async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  const demoPort = Number(process.env.SPATIAL_DEMO_PORT || 18001);
  const sceneId = process.env.SPATIAL_AUDIO_SCENE || "audio-memory-smoke";
  const base = "http://127.0.0.1:" + demoPort;
  const form = new FormData();
  form.append("language", "ja");
  form.append("detection_threshold", "0.25");
  form.append("text_threshold", "0.25");
  form.append("auto_confirm_threshold", "0.60");
  const indexResponse = await fetch(
    base + "/scenes/" + encodeURIComponent(sceneId) + "/audio-memory",
    { method: "POST", body: form }
  );
  const indexed = await indexResponse.json();
  if (!indexResponse.ok) throw new Error("Audio-memory indexing failed: " + JSON.stringify(indexed));
  if (indexed.confirmed_count !== 1 ||
      indexed.confirmed?.[0]?.frame_index !== 10 ||
      indexed.confirmed?.[0]?.entity_name !== "ロボット") {
    throw new Error("Audio-memory real pipeline mismatch: " + JSON.stringify(indexed));
  }

  const pageUrl =
    base + "/?scene=" + encodeURIComponent(sceneId) + "#memory";

  const target = await fetch(
    "http://127.0.0.1:9222/json/new?" + encodeURIComponent(pageUrl),
    { method: "PUT" }
  ).then(r => r.json());
  if (!target?.webSocketDebuggerUrl) {
    throw new Error("Could not open audio-memory smoke tab");
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
      events.push(msg.params.exceptionDetails.exception?.description || msg.params.exceptionDetails.text);
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

  let state = null;
  for (let i = 0; i < 30; i++) {
    await sleep(250);
    state = await evaluate(`(() => ({
      ready: document.readyState,
      screen: document.body.dataset.screen,
      audioStatus: document.getElementById("audioMemoryStatus")?.textContent || "",
      frameIndicator: document.getElementById("frameIndicator")?.textContent || "",
      entities: document.getElementById("entities")?.textContent || ""
    }))()`);
    if (state.screen === "memory" && state.audioStatus.includes("自動確定")) break;
  }

  const report = await fetch(
    base + "/scenes/" + encodeURIComponent(sceneId) + "/audio-memory"
  ).then(async response => {
    const body = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(body));
    return body;
  });

  if (state.screen !== "memory") throw new Error("Memory screen did not open");
  if (!state.audioStatus.includes("自動確定 1件") ||
      !state.audioStatus.includes("確認待ち 0件") ||
      !state.audioStatus.includes("これはロボットです")) {
    throw new Error("Audio-memory summary UI mismatch: " + JSON.stringify(state));
  }
  if (report.confirmed_count !== 1 || report.pending_count !== 0 ||
      report.confirmed?.[0]?.frame_index !== 10 ||
      report.confirmed?.[0]?.entity_name !== "ロボット") {
    throw new Error("Audio-memory report mismatch: " + JSON.stringify(report));
  }
  if (events.length) {
    throw new Error("Unexpected browser exceptions: " + JSON.stringify(events));
  }

  console.log(JSON.stringify({
    state,
    indexed: {
      transcript: indexed.transcript,
      confirmed_count: indexed.confirmed_count,
      pending_count: indexed.pending_count,
      frame_index: indexed.confirmed?.[0]?.frame_index,
      entity_name: indexed.confirmed?.[0]?.entity_name,
    },
    report: {
      transcript: report.transcript,
      segments: report.segments,
      confirmed_count: report.confirmed_count,
      pending_count: report.pending_count,
      confirmed: report.confirmed,
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
