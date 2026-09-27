import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../web/app.js", import.meta.url), "utf8");
function section(start, end) {
  const first = source.indexOf(start);
  const last = source.indexOf(end, first);
  assert.ok(first >= 0 && last > first, "viewer section must exist");
  return source.slice(first, last);
}
// Run the actual viewer functions with controllable network and GPU resources.
function viewer(mobileViewport = false) {
  const requests = [], pending = [], timers = [], objects = [], warnings = [];
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: "", classList: { toggle() {} }, handlers: {},
      addEventListener(name, fn) { this.handlers[name] = fn; },
    });
    return elements.get(id);
  };
  const context = vm.createContext({
    mobileViewport, URLSearchParams, sceneId: "room / test",
    viewerDisposed: false, viewerSuspended: false,
    pointCloud: null, pointCloudLoaded: false, pointCloudLoadPromise: null,
    pointCloudRevision: "revision-1", pointCloudLoadGeneration: 0,
    viewerBounds: {}, viewerRoomFrame: null, spatialModelLoaded: true,
    photoRoom: { group: null, loaded: false },
    spatialShell: { material: {} }, spatialComponentMeshes: new Map(),
    splatLayer: { setVisible(value) { this.visible = value; } },
    scene: { add: item => objects.push(item), remove: item => objects.splice(objects.indexOf(item), 1) },
    renderer: { renderLists: { dispose() {} } },
    document: { hidden: false, body: { dataset: { screen: "room" }, classList: { toggle() {} } }, getElementById: element },
    window: { setTimeout: fn => timers.push(fn) },
    console: { warn: (...args) => warnings.push(args) },
    THREE: {
      PointsMaterial: class {
        constructor(options) { Object.assign(this, options); this.disposed = false; }
        dispose() { this.disposed = true; }
      },
      Points: class { constructor(geometry, material) { Object.assign(this, { geometry, material }); } },
    },
    PLYLoader: class {
      loadAsync(url) {
        requests.push(new URL(url, "http://test"));
        return new Promise((resolve, reject) => pending.push({ resolve, reject }));
      }
    },
  });
  vm.runInContext(
    section("const pointCloudBudgets =", 'document.getElementById("sceneLabel")') +
    section("const viewDisplayState =", "let gridHelper =") +
    section("function disposePointCloud(", "function spatialComponentColor(") +
    section('const togglePointCloudBtn = document.getElementById', 'const toggleMeshBtn = document.getElementById') +
    ";globalThis.displayState = viewDisplayState;", context);
  // These point-cloud cases begin after choosePhotoRoom has selected raw data.
  // The application otherwise correctly defers PLY loading while that check runs.
  vm.runInContext("photoRoomChecking = false;", context);
  const flush = async () => {
    while (timers.length) {
      await timers.shift()();
    }
  };
  const geometry = () => ({
    disposed: false, dispose() { this.disposed = true; },
    computeBoundingBox() {}, computeBoundingSphere() {}, hasAttribute: () => true,
  });
  return { context, requests, pending, objects, warnings, element, flush, geometry };
}

test("desktop preview has a 250k budget and preserves scene/revision", async () => {
  const v = viewer();
  const loading = v.context.loadPointCloud();
  assert.equal(v.requests[0].searchParams.get("max_points"), "250000");
  assert.equal(v.requests[0].searchParams.get("revision"), "revision-1");
  assert.equal(decodeURIComponent(v.requests[0].pathname), "/scenes/room / test/point-cloud");
  v.pending.shift().resolve(v.geometry());
  await loading;
  assert.equal(v.context.pointCloudLoaded, true);
  assert.equal(v.objects.length, 1);
  assert.equal(v.context.pointCloud.material.vertexColors, true);
});

test("point clouds wait for photo-room selection and stay unloaded in photo mode", async () => {
  const v = viewer();
  vm.runInContext("photoRoomChecking = true;", v.context);
  await v.context.loadPointCloud();
  v.context.schedulePointCloudLoad();
  await v.flush();
  assert.equal(v.requests.length, 0);
  vm.runInContext("photoRoomChecking = false; geometryVariant = 'photo-room';", v.context);
  await v.context.loadPointCloud();
  v.context.schedulePointCloudLoad();
  await v.flush();
  assert.equal(v.requests.length, 0);
});

test("mobile uses 100k and full quality explicitly omits the limit", async () => {
  const v = viewer(true);
  let loading = v.context.loadPointCloud();
  assert.equal(v.requests[0].searchParams.get("max_points"), "100000");
  v.pending.shift().resolve(v.geometry());
  await loading;
  v.element("pointCloudQuality").handlers.change({ target: { value: "full" } });
  loading = v.context.loadPointCloud();
  assert.equal(v.requests[1].searchParams.has("max_points"), false);
  v.pending.shift().resolve(v.geometry());
  await loading;
});

test("turning points off frees GPU resources, skips refresh, and keeps mesh opaque", async () => {
  const v = viewer();
  const geometry = v.geometry();
  const loading = v.context.loadPointCloud();
  v.pending.shift().resolve(geometry);
  await loading;
  const material = v.context.pointCloud.material;
  const bounds = v.context.viewerBounds;
  v.element("togglePointCloudBtn").handlers.click();
  assert.equal(geometry.disposed, true);
  assert.equal(material.disposed, true);
  assert.equal(v.objects.length, 0);
  assert.equal(v.context.viewerBounds, bounds);
  assert.equal(v.context.spatialShell.material.opacity, 1);
  await v.context.loadPointCloud();
  v.context.schedulePointCloudLoad();
  await v.flush();
  assert.equal(v.requests.length, 1);
  v.element("togglePointCloudBtn").handlers.click();
  const resumed = v.context.loadPointCloud();
  v.pending.shift().resolve(v.geometry());
  await resumed;
  assert.equal(v.context.pointCloudLoaded, true);
});

test("off during download disposes late geometry and never reloads", async () => {
  const v = viewer();
  const loading = v.context.loadPointCloud();
  v.element("togglePointCloudBtn").handlers.click();
  const geometry = v.geometry();
  v.pending.shift().resolve(geometry);
  await loading;
  await v.flush();
  assert.equal(geometry.disposed, true);
  assert.equal(v.requests.length, 1);
  assert.equal(v.objects.length, 0);
});

test("rapid quality changes discard stale data and request only latest budget", async () => {
  const v = viewer();
  const loading = v.context.loadPointCloud();
  v.context.setPointCloudQuality("full");
  v.context.setPointCloudQuality("light");
  const geometry = v.geometry();
  v.pending.shift().resolve(geometry);
  await loading;
  assert.equal(geometry.disposed, true);
  // Drain scheduling callbacks without waiting on the pending network response.
  const flushing = v.flush();
  await Promise.resolve();
  const latest = v.context.loadPointCloud();
  assert.equal(v.requests.length, 2);
  assert.equal(v.requests[1].searchParams.get("max_points"), "100000");
  v.pending.shift().resolve(v.geometry());
  await Promise.all([latest, flushing]);
  assert.equal(v.objects.length, 1);
});

test("invalid/unchanged quality does not reload or accept inherited keys", () => {
  const v = viewer();
  for (const quality of ["balanced", "unknown", "toString", "__proto__"]) {
    v.context.setPointCloudQuality(quality);
  }
  assert.equal(v.context.pointCloudLoadGeneration, 0);
  assert.equal(v.requests.length, 0);
});

test("suspending during a request discards geometry and allows a later retry", async () => {
  const v = viewer();
  const loading = v.context.loadPointCloud();
  v.context.viewerSuspended = true;
  const geometry = v.geometry();
  v.pending.shift().resolve(geometry);
  await loading;
  assert.equal(geometry.disposed, true);
  assert.equal(v.context.pointCloudLoaded, false);
  v.context.viewerSuspended = false;
  const resumed = v.context.loadPointCloud();
  v.pending.shift().resolve(v.geometry());
  await resumed;
  assert.equal(v.context.pointCloudLoaded, true);
});

test("request failure leaves mesh opaque and permits retry", async () => {
  const v = viewer();
  v.context.updateViewDisplay();
  const loading = v.context.loadPointCloud();
  v.pending.shift().reject(new Error("offline"));
  await loading;
  assert.equal(v.context.pointCloudLoadPromise, null);
  assert.equal(v.context.spatialShell.material.opacity, 1);
  assert.equal(v.warnings.length, 1);
  const retry = v.context.loadPointCloud();
  v.pending.shift().resolve(v.geometry());
  await retry;
  assert.equal(v.context.pointCloudLoaded, true);
});
