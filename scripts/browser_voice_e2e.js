async function main() {
  const demoPort = Number(process.env.SPATIAL_DEMO_PORT || 18001);
  const targets = await fetch("http://127.0.0.1:9222/json").then(r => r.json());
  const target = targets.find(t => t.type === "page" && t.url.includes(`127.0.0.1:${demoPort}`));
  if (!target) throw new Error("Spatial Memory page not found");

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  let nextId = 1;
  const pending = new Map();
  const events = [];

  ws.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const item = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) item.reject(new Error(JSON.stringify(msg.error)));
      else item.resolve(msg.result);
      return;
    }
    if (msg.method === "Runtime.exceptionThrown") {
      events.push({ type: "exception", text: msg.params.exceptionDetails.text });
    }
    if (msg.method === "Runtime.consoleAPICalled") {
      events.push({
        type: "console",
        level: msg.params.type,
        values: (msg.params.args || []).map(a => a.value ?? a.description ?? "")
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
  const evalExpr = async expression => {
    const result = await send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
      userGesture: true,
    });
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.text || "Runtime evaluation failed");
    }
    return result.result.value;
  };

  await send("Runtime.enable");
  await send("Page.enable");

  const result = await evalExpr(`(async () => {
    const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
    const waitUntil = async (fn, timeoutMs = 60000) => {
      const started = performance.now();
      while (performance.now() - started < timeoutMs) {
        const value = fn();
        if (value) return value;
        await sleep(250);
      }
      throw new Error("browser voice E2E timeout");
    };

    document.querySelector('[data-screen-target="memory"]').click();
    await sleep(300);
    document.getElementById("captureFrame").value = "10";
    document.getElementById("capturePrompt").value = "robot";

    const audioResponse = await fetch("/static/test-assets/voice-memory-ja.wav");
    if (!audioResponse.ok) throw new Error("audio fixture fetch failed");
    const audioBlob = await audioResponse.blob();
    const file = new File([audioBlob], "voice-memory-ja.wav", { type: "audio/wav" });
    const dt = new DataTransfer();
    dt.items.add(file);
    document.getElementById("captureAudio").files = dt.files;

    const beforeCards = document.querySelectorAll("#entities .card").length;
    document.getElementById("captureBtn").click();

    await waitUntil(() => {
      const text = document.getElementById("captureStatus").textContent || "";
      return text.includes("記憶しました") || text.includes("failed") || text.includes("失敗");
    });

    const captureStatus = document.getElementById("captureStatus").textContent;
    if (!captureStatus.includes("記憶しました")) {
      throw new Error("voice memory failed: " + captureStatus);
    }

    const afterCards = document.querySelectorAll("#entities .card").length;

    document.getElementById("query").value = "ロボットどこ？";
    document.getElementById("query").dispatchEvent(new Event("input", { bubbles: true }));
    document.getElementById("guideBtn").click();

    await waitUntil(() => {
      const answer = document.getElementById("answer").textContent || "";
      return answer && answer !== "未検索" && !answer.startsWith("記憶しました");
    });

    return {
      captureStatus,
      beforeCards,
      afterCards,
      answer: document.getElementById("answer").textContent,
      query: document.getElementById("query").value,
      captureAudioSize: document.getElementById("captureAudio").files[0]?.size || 0,
      pointCloudCanvasPresent: document.querySelector("#viewport canvas") !== null,
    };
  })()`);

  console.log(JSON.stringify({ result, events: events.slice(-20) }, null, 2));
  ws.close();
}

main().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
