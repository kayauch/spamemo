async function main() {
  const demoPort = Number(process.env.SPATIAL_DEMO_PORT || 18001);
  const targets = await fetch("http://127.0.0.1:9222/json").then(r => r.json());
  const target = targets.find(t => t.type === "page" && t.url.includes(`127.0.0.1:${demoPort}`));
  if (!target) throw new Error("page not found");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  let id = 1;
  const pending = new Map();
  const events = [];
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { const p=pending.get(m.id); pending.delete(m.id); m.error?p.reject(m.error):p.resolve(m.result); return; }
    if (m.method === "Runtime.exceptionThrown") events.push({method:m.method,text:m.params.exceptionDetails.text,exception:m.params.exceptionDetails.exception?.description});
    if (m.method === "Log.entryAdded") events.push({method:m.method,level:m.params.entry.level,text:m.params.entry.text,url:m.params.entry.url});
    if (m.method === "Network.loadingFailed") events.push({method:m.method,error:m.params.errorText,type:m.params.type,url:m.params.blockedReason});
    if (m.method === "Runtime.consoleAPICalled") events.push({method:m.method,type:m.params.type,args:(m.params.args||[]).map(a=>a.value??a.description)});
  };
  await new Promise((res,rej)=>{ws.onopen=res;ws.onerror=rej;});
  const send=(method,params={})=>new Promise((resolve,reject)=>{const n=id++;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});
  await send("Runtime.enable"); await send("Log.enable"); await send("Network.enable"); await send("Page.enable");
  await send("Page.reload",{ignoreCache:true});
  await new Promise(r=>setTimeout(r,6000));
  const state=await send("Runtime.evaluate",{expression:`({scene:document.getElementById("sceneLabel")?.textContent, canvases:document.querySelectorAll("#viewport canvas").length, scripts:[...document.scripts].map(s=>s.src), ready:document.readyState})`,returnByValue:true});
  const resources=await send("Runtime.evaluate",{expression:`performance.getEntriesByType("resource").map(r=>({name:r.name,initiatorType:r.initiatorType,transferSize:r.transferSize}))`,returnByValue:true});
  console.log(JSON.stringify({state:state.result.value,resources:resources.result.value,events},null,2));
  ws.close();
}
main().catch(e=>{console.error(e);process.exit(1);});