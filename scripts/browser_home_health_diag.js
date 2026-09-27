async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  const base = "http://127.0.0.1:18001";
  const target = await fetch(
    "http://127.0.0.1:9222/json/new?" + encodeURIComponent(base + "/"),
    { method: "PUT" }
  ).then(r => r.json());

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
    if (["Runtime.exceptionThrown", "Runtime.consoleAPICalled", "Log.entryAdded"].includes(msg.method)) {
      events.push(msg);
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

  await send("Runtime.enable");
  await send("Log.enable");
  await send("Page.enable");
  await sleep(5000);

  const result = await send("Runtime.evaluate", {
    expression: `JSON.stringify({
      readyState: document.readyState,
      url: location.href,
      healthText: document.getElementById("demoHealth")?.innerText,
      healthHtml: document.getElementById("demoHealth")?.innerHTML,
      scripts: [...document.scripts].map(s => s.src),
      rooms: document.querySelectorAll(".room").length
    })`,
    returnByValue: true,
  });

  console.log(JSON.stringify({
    dom: JSON.parse(result.result.value),
    events,
  }, null, 2));

  ws.close();
  await fetch("http://127.0.0.1:9222/json/close/" + target.id);
}

main().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});

