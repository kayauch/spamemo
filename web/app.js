import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";
import { PLYLoader } from "/static/vendor/PLYLoader.js";
import { SplatLayer } from "/static/splat-layer.js";
import { WalkControls } from "/static/walk-controls.js";
import { PhotoRoomLayer } from "/static/photo-room-layer.js";

const params = new URLSearchParams(window.location.search);
const sceneId = params.get("scene") || "demo-room";
if (params.has("access_token")) {
  params.delete("access_token");
  const cleanQuery = params.toString();
  const cleanUrl = window.location.pathname + (cleanQuery ? "?" + cleanQuery : "") + window.location.hash;
  window.history.replaceState({}, "", cleanUrl);
}
const mobileViewport = window.matchMedia("(max-width: 820px)").matches;
const pointCloudBudgets = { light: 100000, balanced: 250000, full: null };
let pointCloudQuality = mobileViewport ? "light" : "balanced";
let geometryVariant = "raw";
let photoRoomChecking=true, photoRoomChoicePromise=null, photoCameraReady=false;
let appearanceVariant = "current";
document.getElementById("sceneLabel").textContent = "部屋を読み込み中…";

async function safeJson(response, fallbackMsg = "処理に失敗しました") {
  const text = await response.text();
  let json = null;
  try {
    json = JSON.parse(text);
  } catch {}
  if (!response.ok) {
    const detail = json?.detail || text || fallbackMsg;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (json === null) {
    throw new Error("サーバーから不正な応答を受信しました: " + text.slice(0, 100));
  }
  return json;
}

const viewport = document.getElementById("viewport");
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 500);
camera.position.set(7, 7, 8);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, mobileViewport ? 1.5 : 2));
viewport.appendChild(renderer.domElement);
const photoRoom=new PhotoRoomLayer(scene,sceneId,renderer);
renderer.outputColorSpace=THREE.SRGBColorSpace;

let controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.update();

const walker = new WalkControls(camera, renderer.domElement, () => controls, () => setWalking(false));
function setWalking(enabled) {
  if (enabled) {
    if (document.getElementById('geometryVariant').disabled) return;
    set3dSelectMode(false);
    if(geometryVariant==='photo-room'){
      if(!photoRoom.loaded)return;
      const local=photoRoom.frame.room(camera.position);
      const spawn=photoRoom.navigation.findSpawn([local.x,local.y]);if(!spawn)return;
      photoRoom.setOverview(false);camera.position.copy(photoRoom.frame.world(spawn));
      walker.constrainMove=(p,d)=>photoRoom.frame.constrain(photoRoom.navigation,p,d);
      walker.start(viewerBounds?.getSize(new THREE.Vector3()).length()||1);
      walker.speed=1.2*photoRoom.frame.scale;
    }else{
    walker.constrainMove=null;
    document.getElementById('captureViewBtn').click();
    camera.fov=65;
    camera.near=Math.max((spatialModelManifest?.metrics?.scene_diagonal || 1)*.001, .0001);
    camera.updateProjectionMatrix();
    walker.start(spatialModelManifest?.metrics?.scene_diagonal || 1);
    }
  } else walker.stop();
  document.getElementById('walkBtn').textContent=enabled ? '歩行を終了' : '部屋の中を歩く';
  document.getElementById('walkBtn').setAttribute('aria-pressed', String(enabled));
  document.getElementById('walkHelp').hidden=!enabled;
  document.getElementById('photoWalk').textContent=enabled?'歩行を終了':'中を歩く';
  document.getElementById('photoWalk').setAttribute('aria-pressed',String(enabled));
  for(const button of document.querySelectorAll('#photoRoomTools [data-walk-key]'))button.disabled=!enabled;
  document.getElementById('walkModeNote').textContent=geometryVariant==='photo-room'
    ? 'WASD・矢印で歩き、ドラッグで見回します。壁や家具で止まります。Escで終了。'
    : '元データの試験歩行です。Q/Eで高さを調整でき、壁を通り抜けられます。';
}
document.getElementById('walkBtn').addEventListener('click', () => setWalking(!walker.active));

const hemiLight = new THREE.HemisphereLight(0xffffff, 0x263241, 1.25);
scene.add(hemiLight);
const keyLight = new THREE.DirectionalLight(0xffffff, 1.05);
keyLight.position.set(2.5, 4.0, 3.0);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffffff, 0.65);
fillLight.position.set(-2.5, -2.0, -3.0);
scene.add(fillLight);

const viewDisplayState = {
  pointCloud: true,
  mesh: true,
  cad: false,
  pointSize: 0.022,
};

function updateViewDisplay() {
  if(photoRoom.group)photoRoom.group.visible=geometryVariant==='photo-room';
  document.body.classList.toggle('has-photo-room',geometryVariant==='photo-room');
  document.getElementById('photoRoomTools').hidden=geometryVariant!=='photo-room';
  for(const id of ['togglePointCloudBtn','toggleMeshBtn','toggleCadBtn','pointCloudQuality','pointSizeSlider'])
    document.getElementById(id).disabled=geometryVariant==='photo-room';
  if (pointCloud) {
    pointCloud.visible = viewDisplayState.pointCloud;
    if (pointCloud.material) {
      pointCloud.material.size = viewDisplayState.pointSize;
      pointCloud.material.needsUpdate = true;
    }
  }
  if (spatialShell) {
    spatialShell.visible = viewDisplayState.mesh && geometryVariant !== "2dgs";
    if (spatialShell.material) {
      if (pointCloud?.visible && viewDisplayState.mesh) {
        spatialShell.material.transparent = true;
        spatialShell.material.opacity = 0.45;
        spatialShell.material.depthWrite = false;
      } else {
        spatialShell.material.transparent = false;
        spatialShell.material.opacity = 1.0;
        spatialShell.material.depthWrite = true;
      }
      spatialShell.material.needsUpdate = true;
    }
  }
  for (const mesh of spatialComponentMeshes.values()) {
    mesh.visible = viewDisplayState.cad;
  }
  splatLayer.setVisible(["2dgs", "fusion"].includes(geometryVariant) && viewDisplayState.mesh);

  const pcBtn = document.getElementById("togglePointCloudBtn");
  const meshBtn = document.getElementById("toggleMeshBtn");
  const cadBtn = document.getElementById("toggleCadBtn");
  const slider = document.getElementById("pointSizeSlider");
  const valLabel = document.getElementById("pointSizeValue");
  if (pcBtn) {
    pcBtn.classList.toggle("active", viewDisplayState.pointCloud);
    pcBtn.classList.toggle("secondary", !viewDisplayState.pointCloud);
  }
  if (meshBtn) {
    meshBtn.classList.toggle("active", viewDisplayState.mesh);
    meshBtn.classList.toggle("secondary", !viewDisplayState.mesh);
  }
  if (cadBtn) {
    cadBtn.classList.toggle("active", viewDisplayState.cad);
    cadBtn.classList.toggle("secondary", !viewDisplayState.cad);
  }
  if (slider) slider.value = String(viewDisplayState.pointSize);
  if (valLabel) valLabel.textContent = Number(viewDisplayState.pointSize).toFixed(3);
}

let gridHelper = null;
let axesHelper = null;
let viewerBounds = null;

function disposeSceneHelpers() {
  if (gridHelper) {
    scene.remove(gridHelper);
    gridHelper.geometry?.dispose();
    const materials = Array.isArray(gridHelper.material)
      ? gridHelper.material
      : [gridHelper.material];
    materials.forEach(material => material?.dispose());
    gridHelper = null;
  }
  if (axesHelper) {
    scene.remove(axesHelper);
    axesHelper.geometry?.dispose();
    axesHelper.material?.dispose();
    axesHelper = null;
  }
}

function robustBoundsFromGeometry(geometry, maxSamples = 50000) {
  const positions = geometry.getAttribute("position");
  if (!positions || positions.count < 1) return null;
  const step = Math.max(1, Math.ceil(positions.count / maxSamples));
  const xs = [];
  const ys = [];
  const zs = [];
  for (let i = 0; i < positions.count; i += step) {
    const x = positions.getX(i);
    const y = positions.getY(i);
    const z = positions.getZ(i);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    xs.push(x);
    ys.push(y);
    zs.push(z);
  }
  if (!xs.length) return null;
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  zs.sort((a, b) => a - b);
  const quantile = (values, q) => {
    const index = Math.min(
      values.length - 1,
      Math.max(0, Math.round((values.length - 1) * q))
    );
    return values[index];
  };
  return new THREE.Box3(
    new THREE.Vector3(
      quantile(xs, 0.01),
      quantile(ys, 0.01),
      quantile(zs, 0.01)
    ),
    new THREE.Vector3(
      quantile(xs, 0.99),
      quantile(ys, 0.99),
      quantile(zs, 0.99)
    )
  );
}

function updateSceneHelpers(bounds, roomFrame = null) {
  disposeSceneHelpers();
  if (!bounds || bounds.isEmpty()) return;
  const size = bounds.getSize(new THREE.Vector3());
  const center = bounds.getCenter(new THREE.Vector3());
  const maxSpan = Math.max(size.x, size.y, size.z, 0.01);
  const horizontalSpan = Math.max(size.x, size.z, maxSpan * 0.4);
  const gridSize = Math.max(0.25, horizontalSpan * 1.35);
  const divisions = Math.max(
    6,
    Math.min(16, Math.round(gridSize / Math.max(maxSpan / 10, 0.05)))
  );
  gridHelper = new THREE.GridHelper(gridSize, divisions);
  gridHelper.visible = document.getElementById("showGrid")?.checked ?? true;
  gridHelper.userData.size = gridSize;
  gridHelper.userData.divisions = divisions;

  if (roomFrame?.snap_enabled) {
    const axisX = vec(roomFrame.axis_x).normalize();
    const axisY = vec(roomFrame.axis_y_up).normalize();
    const axisZ = vec(roomFrame.axis_z).normalize();
    const basis = new THREE.Matrix4().makeBasis(axisX, axisY, axisZ);
    gridHelper.setRotationFromMatrix(basis);
    const origin = vec(roomFrame.origin);
    gridHelper.position.copy(origin).addScaledVector(
      axisY,
      -Math.max(maxSpan * 0.006, 0.002)
    );
    gridHelper.userData.roomAligned = true;
    gridHelper.userData.up = { x: axisY.x, y: axisY.y, z: axisY.z };
  } else {
    gridHelper.position.set(
      center.x,
      bounds.min.y - Math.max(size.y * 0.015, 0.003),
      center.z
    );
    gridHelper.userData.roomAligned = false;
  }
  scene.add(gridHelper);

  axesHelper = new THREE.AxesHelper(Math.max(0.08, maxSpan * 0.12));
  if (roomFrame?.snap_enabled) {
    const axisX = vec(roomFrame.axis_x).normalize();
    const axisY = vec(roomFrame.axis_y_up).normalize();
    const axisZ = vec(roomFrame.axis_z).normalize();
    axesHelper.setRotationFromMatrix(
      new THREE.Matrix4().makeBasis(axisX, axisY, axisZ)
    );
    axesHelper.position.copy(vec(roomFrame.origin));
  } else {
    axesHelper.position.set(bounds.min.x, bounds.min.y, bounds.min.z);
  }
  scene.add(axesHelper);
}

const entityGroup = new THREE.Group();
scene.add(entityGroup);
let entityMeshes = new Map();
let routeLine = null;
let pointCloud = null;
let pointCloudLoaded = false;
let pointCloudLoadPromise = null;
let pointCloudRevision = "";
let pointCloudLoadGeneration = 0;
const spatialModelGroup = new THREE.Group();
scene.add(spatialModelGroup);
const splatLayer = new SplatLayer(scene);
let spatialShell = null;
let spatialComponentMeshes = new Map();
let spatialModelManifest = null;
let spatialModelLoaded = false;
let spatialModelLoadPromise = null;
let spatialModelRevision = "";
let spatialModelPath = "";
let spatialModelLoadGeneration = 0;
let spatialRenderMode = "legacy";
let viewerRoomFrame = null;
let animationFrameId = null;
let viewerDisposed = false;
let viewerSuspended = false;
let select3dMode = false;
let selected3dPosition = null;
let selected3dSource = null;
let selectionMarker = null;
const selectionRaycaster = new THREE.Raycaster();
const selectionPointer = new THREE.Vector2();
let selectionPointerDown = null;

function resize() {
  const w = viewport.clientWidth;
  const h = viewport.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function animate() {
  if (viewerDisposed || document.hidden) {
    animationFrameId = null;
    return;
  }
  animationFrameId = requestAnimationFrame(animate);
  if (walker.active) walker.update(performance.now());
  else controls.update();
  renderer.render(scene, camera);
}

function resumeAnimation() {
  if (!viewerDisposed && animationFrameId === null && !document.hidden) animate();
}
resumeAnimation();

function vec(p) {
  return new THREE.Vector3(p.x, p.y, p.z);
}

function clearEntities() {
  for (const mesh of entityMeshes.values()) {
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  entityMeshes.clear();
  entityGroup.clear();
}

function renderEntities(entities, focusId = null) {
  clearEntities();
  const list = document.getElementById("entities");
  list.innerHTML = "";
  const parentSelect = document.getElementById("parentEntity");
  const previousParent = parentSelect ? parentSelect.value : "";
  if (parentSelect) parentSelect.innerHTML = "";
  const orderedEntities = [...entities].sort((a, b) => {
    const rank = entity => entity.kind === "container" ? 0 :
      entity.kind === "furniture" ? 1 : 2;
    return rank(a) - rank(b);
  });
  for (const entity of entities) {
    const focused = entity.id === focusId;
    const geometry = new THREE.SphereGeometry(geometryVariant==='photo-room' ? (focused?.09:.05)*photoRoom.frame.scale : (focused?.20:.12),20,12);
    const material = new THREE.MeshBasicMaterial({
      color: focused ? 0xff6600 : 0x34a853,
      depthTest: !focused,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(vec(entity.position));
    mesh.userData.entity = entity;
    mesh.renderOrder=focused?1000:1;
    entityGroup.add(mesh);
    entityMeshes.set(entity.id, mesh);

    const div = document.createElement("div");
    div.className = "card";
    const observationCount = Number(entity.metadata?.observation_count || 1);
    div.textContent = entity.name + "  (" +
      entity.position.x.toFixed(1) + ", " +
      entity.position.y.toFixed(1) + ", " +
      entity.position.z.toFixed(1) + ")  [" +
      entity.source + ", " + observationCount + " obs]";
    list.appendChild(div);
  }
  if (parentSelect) {
    for (const entity of orderedEntities) {
      const option = document.createElement("option");
      option.value = entity.id;
      option.textContent = entity.name + " [" + entity.kind + "]";
      parentSelect.appendChild(option);
    }
    if ([...parentSelect.options].some(option => option.value === previousParent)) {
      parentSelect.value = previousParent;
    }
  }
}

function renderRoute(points) {
  if (routeLine) {
    scene.remove(routeLine);
    routeLine.geometry.dispose();
    routeLine.material.dispose();
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map(vec));
  const material = new THREE.LineBasicMaterial({ color: 0x4da3ff });
  routeLine = new THREE.Line(geometry, material);
  scene.add(routeLine);
}

function fitCameraToBounds(bounds, padding = 1.12, roomFrame = null) {
  if (!bounds || bounds.isEmpty()) return;
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const radius = Math.max(size.length() * 0.5, 0.05);
  const verticalFov = THREE.MathUtils.degToRad(camera.fov);
  const horizontalFov = 2 * Math.atan(
    Math.tan(verticalFov * 0.5) * Math.max(camera.aspect, 0.1)
  );
  const limitingFov = Math.max(
    Math.min(verticalFov, horizontalFov),
    THREE.MathUtils.degToRad(20)
  );
  const distance = Math.max(
    radius * padding / Math.sin(limitingFov * 0.5),
    radius * 1.25,
    0.15
  );

  let direction = new THREE.Vector3(1.0, 0.62, 1.0).normalize();
  if (roomFrame?.snap_enabled) {
    const axisX = vec(roomFrame.axis_x).normalize();
    const axisY = vec(roomFrame.axis_y_up).normalize();
    const axisZ = vec(roomFrame.axis_z).normalize();
    direction = axisX
      .clone()
      .addScaledVector(axisY, 0.62)
      .add(axisZ)
      .normalize();
  }

  camera.position.copy(center).addScaledVector(direction, distance);
  camera.near = Math.max(distance / 500, 0.005);
  camera.far = Math.max(distance * 12, radius * 20, 10);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function fitCameraToObject(object, padding = 1.12) {
  const box = new THREE.Box3().setFromObject(object);
  fitCameraToBounds(box, padding);
}

function disposePointCloud(preserveHelpers = false) {
  pointCloudLoadGeneration += 1;
  if (pointCloud) {
    scene.remove(pointCloud);
    pointCloud.geometry?.dispose();
    pointCloud.material?.dispose();
    pointCloud = null;
  }
  pointCloudLoaded = false;
  if (!preserveHelpers && !spatialModelLoaded) {
    viewerBounds = null;
    disposeSceneHelpers();
  }
  renderer.renderLists?.dispose();
}

async function loadPointCloud() {
  if(photoRoomChecking || geometryVariant==='photo-room')return;
  if (viewerDisposed || viewerSuspended || document.hidden || !viewDisplayState.pointCloud || pointCloudLoaded) return;
  if (pointCloudLoadPromise) return pointCloudLoadPromise;

  const generation = pointCloudLoadGeneration;
  const queryParams = new URLSearchParams();
  queryParams.set("variant", ["texture", "2dgs", "fusion"].includes(geometryVariant) ? "hybrid" : geometryVariant);
  const maxPoints = pointCloudBudgets[pointCloudQuality];
  if (maxPoints != null) queryParams.set("max_points", String(maxPoints));
  if (pointCloudRevision) queryParams.set("revision", pointCloudRevision);
  const query = queryParams.toString() ? "?" + queryParams.toString() : "";
  pointCloudLoadPromise = (async () => {
    try {
      const loader = new PLYLoader();
      const geometry = await loader.loadAsync(
        "/scenes/" + encodeURIComponent(sceneId) + "/point-cloud" + query
      );
      if (viewerDisposed || viewerSuspended || document.hidden || !viewDisplayState.pointCloud || generation !== pointCloudLoadGeneration) {
        geometry.dispose();
        return;
      }
      geometry.computeBoundingBox();
      geometry.computeBoundingSphere();
      const hasColors = geometry.hasAttribute("color");
      const material = new THREE.PointsMaterial({
        size: viewDisplayState.pointSize,
        sizeAttenuation: true,
        vertexColors: hasColors,
        color: hasColors ? 0xffffff : 0xbfc7d5,
      });
      pointCloud = new THREE.Points(geometry, material);
      pointCloud.frustumCulled = false;
      pointCloud.visible = viewDisplayState.pointCloud;
      scene.add(pointCloud);
      pointCloudLoaded = true;
      if (!viewerBounds) {
        viewerBounds = robustBoundsFromGeometry(geometry) || geometry.boundingBox?.clone() || null;
        updateSceneHelpers(viewerBounds, viewerRoomFrame);
        if (viewerBounds) fitCameraToBounds(viewerBounds, 1.12, viewerRoomFrame);
        else fitCameraToObject(pointCloud);
      }
      updateViewDisplay();
    } catch (error) {
      console.warn("Point cloud unavailable:", error);
    } finally {
      const staleLoad = generation !== pointCloudLoadGeneration;
      pointCloudLoadPromise = null;
      if (
        staleLoad &&
        viewDisplayState.pointCloud &&
        !pointCloudLoaded &&
        !viewerDisposed &&
        !viewerSuspended &&
        !document.hidden &&
        ["room", "find"].includes(document.body.dataset.screen)
      ) {
        window.setTimeout(() => schedulePointCloudLoad(), 0);
      }
    }
  })();
  return pointCloudLoadPromise;
}

function schedulePointCloudLoad() {
  if(photoRoomChecking || geometryVariant==='photo-room')return;
  if (viewerDisposed || viewerSuspended || document.hidden || !viewDisplayState.pointCloud || pointCloudLoaded || pointCloudLoadPromise) return;
  const run = () => loadPointCloud().catch(error => console.warn(error));
  if (mobileViewport && "requestIdleCallback" in window) {
    window.requestIdleCallback(run, { timeout: 1200 });
  } else {
    window.setTimeout(run, mobileViewport ? 250 : 0);
  }
}

function showPhotoError(error){
  document.getElementById('photoRoomState').textContent='写真付きモデルを読み込めませんでした。ページを再読み込みしてください。';
  console.warn('Photo room:',error);
}
async function choosePhotoRoom(){
  if(photoRoomChoicePromise)return photoRoomChoicePromise;
  photoRoomChoicePromise=(async()=>{
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);
    try{
      const jobResponse=await fetch('/scenes/'+encodeURIComponent(sceneId)+'/solid-room-job',{signal:controller.signal});
      if(!jobResponse.ok)return;
      const job=await jobResponse.json();
      if(!job.result_url&&!job.previous_result_url)return;
      const result=new URL(job.result_url||job.previous_result_url,location.origin);
      const run=result.searchParams.get('run'),name=result.searchParams.get('result');
      if(result.origin!==location.origin||result.pathname!=='/static/solid-room.html'||![run,name].every(v=>v&&/^[a-zA-Z0-9_-]+$/.test(v)))return;
      const base=new URL('/static/generation-reviews/'+run+'/'+name+'/',location.origin);
      const response=await fetch(new URL('room.json',base),{signal:controller.signal});
      if(!response.ok)throw Error('モデルの情報を取得できませんでした');
      photoRoom.configure(await response.json(),base.href);
      geometryVariant='photo-room';
      const option=document.querySelector('#geometryVariant option[value="photo-room"]');option.disabled=false;
      document.getElementById('geometryVariant').value=geometryVariant;
      viewDisplayState.pointCloud=false;viewDisplayState.mesh=true;viewDisplayState.cad=false;
      updateViewDisplay();
    }catch(error){showPhotoError(error);}
    finally{
      clearTimeout(timer);photoRoomChecking=false;
      if(geometryVariant!=='photo-room')document.getElementById('photoRoomState').textContent='元の3Dデータを表示中 · 写真付きモデルは「部屋を見る」から作成できます';
      setWorkspaceScreen(screenFromHash());
    }
  })();
  return photoRoomChoicePromise;
}
async function loadPhotoRoom(){
  if(viewerDisposed||viewerSuspended||document.hidden||!photoRoom.data)return false;
  if(photoRoom.loaded)return true;
  const loaded=await photoRoom.load();
  if(!loaded||viewerDisposed||viewerSuspended||geometryVariant!=='photo-room')return false;
  viewerBounds=photoRoom.bounds;disposeSceneHelpers();
  camera.near=.025*photoRoom.frame.scale;camera.far=Math.max(100*photoRoom.frame.scale,10);camera.fov=65;camera.updateProjectionMatrix();
  if(!photoCameraReady){
    camera.up.copy(photoRoom.frame.up);controls.dispose();controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
    photoRoom.inside(camera,controls);photoCameraReady=true;
    for(const axis of ['x','y','z'])document.getElementById('s'+axis).value=String(camera.position[axis]);
  }
  document.getElementById('photoRoomState').textContent='実写テクスチャ付きモデル · 登録した位置と同じ座標で表示';
  document.getElementById('geometryStatus').textContent='壁・家具・写真は部屋モデルと共通です。';
  document.getElementById('captureViewBtn').disabled=true;
  updateViewDisplay();resize();return true;
}
for(const [id,mode] of [['photoInside','inside'],['photoOverview','overview']]){
  document.getElementById(id).onclick=()=>{if(walker.active)setWalking(false);photoRoom[mode](camera,controls);};
}
document.getElementById('photoWalk').onclick=()=>setWalking(!walker.active);

function spatialComponentColor(kind) {
  const colors = {
    floor: 0xb8bec7,
    ceiling: 0xd1d5db,
    wall: 0xc4c9d0,
    horizontal_surface: 0xaeb6c0,
    sloped_surface: 0xa8b0bb,
    object_region: 0x9da8b5,
  };
  return colors[kind] ?? 0xaeb6c0;
}

function spatialComponentOpacity(component) {
  if (component.kind === "ceiling") return 0.24;
  if (component.kind === "wall") return 0.58;
  if (component.kind === "floor") return 0.78;
  if (component.kind === "object_region") return 0.86;
  return 0.72;
}

function disposeSpatialModel() {
  splatLayer.clear();
  spatialModelLoadGeneration += 1;
  for (const child of [...spatialModelGroup.children]) {
    spatialModelGroup.remove(child);
    child.geometry?.dispose();
    if (Array.isArray(child.material)) {
      child.material.forEach(material => { material?.map?.dispose(); material?.dispose(); });
    } else {
      child.material?.map?.dispose();
      child.material?.dispose();
    }
  }
  spatialShell = null;
  spatialComponentMeshes.clear();
  spatialModelManifest = null;
  spatialModelLoaded = false;
  spatialModelRevision = "";
  spatialRenderMode = "legacy";
  viewerRoomFrame = null;
  if (!pointCloudLoaded) {
    viewerBounds = null;
    disposeSceneHelpers();
  }
  renderer.renderLists?.dispose();
}

function buildSpatialComponentMesh(component) {
  const center = vec(component.center);
  const vertices = component.vertices || [];
  const positions = new Float32Array(vertices.length * 3);
  vertices.forEach((vertex, index) => {
    positions[index * 3] = Number(vertex[0]) - center.x;
    positions[index * 3 + 1] = Number(vertex[1]) - center.y;
    positions[index * 3 + 2] = Number(vertex[2]) - center.z;
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex((component.triangles || []).flat());
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  const opacity = spatialComponentOpacity(component);
  const material = new THREE.MeshStandardMaterial({
    color: spatialComponentColor(component.kind),
    transparent: opacity < 0.999,
    opacity,
    depthWrite: opacity >= 0.72,
    side: THREE.DoubleSide,
    flatShading: true,
    roughness: 0.96,
    metalness: 0.0,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(center);
  mesh.userData.component = component;
  mesh.userData.regularized =
    component.source === "structural_regularized" ||
    component.source === "object_regularized";
  mesh.renderOrder = component.kind === "object_region" ? 4 : 3;
  return mesh;
}

async function loadSpatialModel() {
  if(photoRoomChecking)return false;
  if(geometryVariant==='photo-room')return loadPhotoRoom();
  if (viewerDisposed || viewerSuspended || document.hidden || spatialModelLoaded) {
    return spatialModelLoaded;
  }
  if (spatialModelLoadPromise) return spatialModelLoadPromise;

  const generation = spatialModelLoadGeneration;
  spatialModelLoadPromise = (async () => {
    try {
      const response = await fetch(
        "/scenes/" + encodeURIComponent(sceneId) + "/spatial-model"
      );
      if (response.status === 404) return false;
      const manifest = await safeJson(response, "Spatial Modelの読み込みに失敗しました");
      if (
        viewerDisposed ||
        viewerSuspended ||
        document.hidden ||
        generation !== spatialModelLoadGeneration
      ) {
        return false;
      }

      if (
        spatialModelLoaded &&
        spatialModelRevision &&
        manifest.revision !== spatialModelRevision
      ) {
        disposeSpatialModel();
      }
      spatialModelRevision = manifest.revision || "";
      spatialModelManifest = manifest;
      const hybridOption = document.querySelector('#geometryVariant option[value="hybrid"]');
      if (hybridOption) hybridOption.disabled = !manifest.hybrid;
      const textureOption = document.querySelector('#geometryVariant option[value="texture"]');
      if (textureOption) textureOption.disabled = !manifest.texture;
      const splatOption = document.querySelector('#geometryVariant option[value="2dgs"]');
      if (splatOption) splatOption.disabled = !manifest.appearance_2dgs;
      const fusionOption = document.querySelector('#geometryVariant option[value="fusion"]');
      if (fusionOption) fusionOption.disabled = !manifest.appearance_2dgs || !manifest.fusion;
      document.getElementById("captureViewBtn").disabled = !manifest.appearance_2dgs?.capture_views?.length;
      viewerRoomFrame = manifest.room_frame || null;

      const structuralComponents =
        manifest.structural_layout?.enabled
          ? (manifest.structural_layout.components || [])
          : [];
      const objectComponents =
        manifest.object_layout?.enabled
          ? (manifest.object_layout.components || [])
          : [];
      const regularizedComponents = [
        ...structuralComponents,
        ...objectComponents,
      ];
      const componentFirst =
        viewerRoomFrame?.snap_enabled &&
        regularizedComponents.length >= 3;
      spatialRenderMode = componentFirst ? "component-first" : "legacy";

      const loader = new PLYLoader();
      const shellUrl =
        (manifest.shell?.url ||
          ("/scenes/" + encodeURIComponent(sceneId) + "/spatial-model/shell")) +
        "?variant=" + (geometryVariant === "2dgs" ? (manifest.hybrid ? "hybrid" : "raw") : geometryVariant) + "&revision=" + encodeURIComponent((geometryVariant === "fusion" ? manifest.fusion?.revision : manifest.revision) || "");
      let geometry;
      let photoTexture = null;
      if (geometryVariant === "texture") {
        const base = "/scenes/" + encodeURIComponent(sceneId) + "/spatial-model/texture/";
        const revision = "?revision=" + encodeURIComponent(manifest.texture?.revision || "");
        const data = await safeJson(await fetch(base + "geometry" + revision), "写真の形状を読み込めませんでした");
        // Load the image first so an image failure cannot leak allocated geometry.
        photoTexture = await new THREE.TextureLoader().loadAsync(base + "atlas" + revision);
        photoTexture.colorSpace = THREE.SRGBColorSpace;
        photoTexture.generateMipmaps = false;
        photoTexture.minFilter = THREE.LinearFilter;
        geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(data.positions, 3));
        geometry.setAttribute("uv", new THREE.Float32BufferAttribute(data.uv, 2));
        // PLYLoader also converts saved sRGB vertex colors into linear space.
        const linearColors = data.colors.map(value => value <= 0.04045
          ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4));
        geometry.setAttribute("color", new THREE.Float32BufferAttribute(linearColors, 3));
      } else {
        geometry = await loader.loadAsync(shellUrl);
      }
      if (
        viewerDisposed ||
        viewerSuspended ||
        document.hidden ||
        generation !== spatialModelLoadGeneration
      ) {
        geometry.dispose();
        photoTexture?.dispose();
        return false;
      }

      if (!geometry.hasAttribute("normal")) geometry.computeVertexNormals();
      geometry.computeBoundingBox();
      geometry.computeBoundingSphere();
      const hasColors = geometry.hasAttribute("color");
      const shellMaterial = geometryVariant === "fusion" ? new THREE.MeshBasicMaterial({
        color: hasColors ? 0xffffff : 0xb8bec8, vertexColors: hasColors,
        side: THREE.DoubleSide, depthWrite: true,
        // Move only rasterized depth slightly back to avoid coplanar clipping.
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      }) : photoTexture ? new THREE.MeshBasicMaterial({
        map: photoTexture, vertexColors: hasColors, side: THREE.DoubleSide,
      }) : new THREE.MeshStandardMaterial({
        color: hasColors ? 0xffffff : 0xb8bec8,
        vertexColors: hasColors,
        transparent: false,
        opacity: 1.0,
        depthWrite: true,
        flatShading: false,
        roughness: 0.85,
        metalness: 0.05,
        side: THREE.DoubleSide,
      });
      spatialShell = new THREE.Mesh(geometry, shellMaterial);
      spatialShell.userData.spatialShell = true;
      spatialShell.userData.supportLayer = false;
      spatialShell.renderOrder = 0;
      spatialShell.visible = viewDisplayState.mesh;
      spatialModelGroup.add(spatialShell);

      if (["2dgs", "fusion"].includes(geometryVariant)) {
        spatialShell.visible = geometryVariant === "fusion";
        const appearance = appearanceVariant === 'baseline' ? manifest.appearance_2dgs_baseline : manifest.appearance_2dgs;
        const loaded = await splatLayer.load("/scenes/" + encodeURIComponent(sceneId) +
          "/spatial-model/appearance-2dgs?variant=" + appearanceVariant + "&revision=" + encodeURIComponent(appearance?.revision || ""));
        if (!loaded || generation !== spatialModelLoadGeneration) return false;
      }

      const displayComponents = manifest.components || regularizedComponents || [];
      for (const component of displayComponents) {
        const mesh = buildSpatialComponentMesh(component);
        mesh.visible = viewDisplayState.cad;
        spatialModelGroup.add(mesh);
        spatialComponentMeshes.set(component.id, mesh);
      }

      spatialModelLoaded = true;
      if (!document.getElementById('geometryVariant').disabled) {
        document.getElementById('appearanceVariant').disabled = !manifest.appearance_2dgs_baseline;
      }
      updateViewDisplay();

      if (manifest.bounds?.min && manifest.bounds?.max) {
        viewerBounds = new THREE.Box3(
          vec(manifest.bounds.min),
          vec(manifest.bounds.max)
        );
      } else {
        viewerBounds =
          robustBoundsFromGeometry(geometry) ||
          geometry.boundingBox?.clone() ||
          null;
      }
      updateSceneHelpers(viewerBounds, viewerRoomFrame);
      if (viewerBounds) fitCameraToBounds(viewerBounds, 1.12, viewerRoomFrame);
      else fitCameraToObject(spatialShell);
      return true;
    } catch (error) {
      console.warn("Spatial model unavailable:", error);
      return false;
    } finally {
      spatialModelLoadPromise = null;
    }
  })();
  return spatialModelLoadPromise;
}

function scheduleSpatialModelLoad() {
  if(photoRoomChecking)return;
  if(geometryVariant==='photo-room'){loadPhotoRoom().catch(showPhotoError);return;}
  if (
    viewerDisposed ||
    viewerSuspended ||
    document.hidden ||
    spatialModelLoaded ||
    spatialModelLoadPromise
  ) {
    return;
  }
  const run = () => loadSpatialModel().catch(error => console.warn(error));
  if (mobileViewport && "requestIdleCallback" in window) {
    window.requestIdleCallback(run, { timeout: 1000 });
  } else {
    window.setTimeout(run, mobileViewport ? 120 : 0);
  }
}

function disposeSelectionMarker() {
  if (!selectionMarker) return;
  scene.remove(selectionMarker);
  selectionMarker.geometry?.dispose();
  selectionMarker.material?.dispose();
  selectionMarker = null;
}

function showSelectionMarker(position) {
  disposeSelectionMarker();
  const geometry = new THREE.SphereGeometry(geometryVariant==='photo-room'?.08*photoRoom.frame.scale:.16,20,12);
  const material = new THREE.MeshBasicMaterial({
    color: 0xffcc00,
    wireframe: true,
    depthTest: false,
  });
  selectionMarker = new THREE.Mesh(geometry, material);
  selectionMarker.position.copy(position);
  selectionMarker.renderOrder = 999;
  scene.add(selectionMarker);
}

function clear3dSelection({ keepMode = true } = {}) {
  selected3dPosition = null;
  selected3dSource = null;
  disposeSelectionMarker();
  const sheet = document.getElementById("selectionSheet");
  const nameInput = document.getElementById("selectionName");
  if (sheet) sheet.hidden = true;
  if (nameInput) nameInput.value = "";
  if (!keepMode) {
    select3dMode = false;
    document.body.classList.remove("select3d-active");
    const button = document.getElementById("select3dBtn");
    const hint = document.getElementById("select3dHint");
    if (button) {
      button.textContent = "＋ 3Dから物を登録";
      button.classList.add("secondary");
    }
    if (hint) hint.hidden = true;
  }
}

function set3dSelectMode(enabled) {
  select3dMode = enabled;
  document.body.classList.toggle("select3d-active", enabled);
  const button = document.getElementById("select3dBtn");
  const hint = document.getElementById("select3dHint");
  if (button) {
    button.textContent = enabled ? "選択モードを終了" : "＋ 3Dから物を登録";
    button.classList.toggle("secondary", !enabled);
  }
  if (hint) hint.hidden = !enabled;
  if (!enabled) clear3dSelection({ keepMode: false });
}

function selectedPointCluster(intersection) {
  if (!pointCloud || intersection.index == null) return intersection.point.clone();
  const geometry = pointCloud.geometry;
  const positions = geometry.getAttribute("position");
  if (!positions) return intersection.point.clone();
  if (!geometry.boundingSphere) geometry.computeBoundingSphere();
  const sceneRadius = geometry.boundingSphere?.radius || 1;
  const radius = THREE.MathUtils.clamp(sceneRadius * 0.025, 0.06, 0.35);
  const localCenter = pointCloud.worldToLocal(intersection.point.clone());
  const radiusSq = radius * radius;
  const centroid = new THREE.Vector3();
  let count = 0;

  for (let i = 0; i < positions.count; i++) {
    const dx = positions.getX(i) - localCenter.x;
    const dy = positions.getY(i) - localCenter.y;
    const dz = positions.getZ(i) - localCenter.z;
    if (dx * dx + dy * dy + dz * dz <= radiusSq) {
      centroid.x += positions.getX(i);
      centroid.y += positions.getY(i);
      centroid.z += positions.getZ(i);
      count += 1;
    }
  }
  if (!count) return intersection.point.clone();
  centroid.multiplyScalar(1 / count);
  return pointCloud.localToWorld(centroid);
}

function select3dAtPointer(clientX, clientY) {
  if (!select3dMode || !["room","memory"].includes(document.body.dataset.screen)) return;
  const rect = renderer.domElement.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  selectionPointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  selectionPointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  selectionRaycaster.setFromCamera(selectionPointer, camera);

  const pointRadius = pointCloud?.geometry?.boundingSphere?.radius || 1;
  selectionRaycaster.params.Points.threshold = THREE.MathUtils.clamp(pointRadius * 0.012, 0.025, 0.18);

  const entityHits = selectionRaycaster.intersectObjects([...entityMeshes.values()], false);
  let position = null;
  let source = null;

  if (entityHits.length) {
    const mesh = entityHits[0].object;
    position = mesh.getWorldPosition(new THREE.Vector3());
    source = { type: "entity", entity: mesh.userData.entity };
  }
  if(!source && geometryVariant==='photo-room'){
    const hit=photoRoom.hit(selectionRaycaster);
    if(hit){position=hit.point.clone();source={type:'photo-room',surfaceId:hit.object.userData.id};}
  }
  if (!source && pointCloud && pointCloud.visible) {
    const pointHits = selectionRaycaster.intersectObject(pointCloud, false);
    if (pointHits.length) {
      position = selectedPointCluster(pointHits[0]);
      source = { type: "point-cloud", pointIndex: pointHits[0].index ?? null };
    }
  }
  if (!source && spatialShell && (spatialShell.visible || (["2dgs", "fusion"].includes(geometryVariant) && viewDisplayState.mesh))) {
    const shellHits = selectionRaycaster.intersectObject(spatialShell, false);
    if (shellHits.length) {
      position = shellHits[0].point.clone();
      source = {
        type: "shell",
        faceIndex: shellHits[0].faceIndex ?? null,
      };
    }
  }
  if (!source && viewDisplayState.cad && spatialComponentMeshes.size) {
    const visibleComponents = [...spatialComponentMeshes.values()].filter(mesh => mesh.visible);
    if (visibleComponents.length) {
      const componentHits = selectionRaycaster.intersectObjects(visibleComponents, false);
      if (componentHits.length) {
        const hit = componentHits[0];
        position = hit.point ? hit.point.clone() : hit.object.getWorldPosition(new THREE.Vector3());
        source = {
          type: "component",
          component: hit.object.userData.component,
        };
      }
    }
  }

  const status = document.getElementById("selectionStatus");
  if (!position || !source) {
    if (status) status.textContent = "選択できませんでした。点や物体の近くをタップしてください。";
    return;
  }

  selected3dPosition = position.clone();
  selected3dSource = source;
  showSelectionMarker(position);
  const sheet = document.getElementById("selectionSheet");
  const input = document.getElementById("selectionName");
  if (sheet) sheet.hidden = false;
  if (source.type === "entity" && source.entity) {
    if (status) status.textContent = "既存の物体を選択: " + source.entity.name;
    if (input) input.value = source.entity.name;
  } else if (source.type === "component" && source.component) {
    if (status) {
      status.textContent =
        "モデルcomponentを選択: " +
        source.component.kind +
        " / " +
        source.component.id;
    }
    if (input) {
      input.value = "";
      input.placeholder = "例: 机 / 棚 / 充電器";
    }
  } else {
    if (status) status.textContent = "3Dモデル上の位置を選択しました";
    if (input) {
      input.value = "";
      input.placeholder = "例: 充電器";
    }
  }
  input?.focus({ preventScroll: true });
}

async function save3dSelection() {
  if (!selected3dPosition) throw new Error("先に3D空間の物をタップしてください");
  const name = document.getElementById("selectionName").value.trim();
  if (!name) throw new Error("この物の名前を入力してください");
  const button = document.getElementById("save3dSelectionBtn");
  const status = document.getElementById("selectionStatus");
  button.disabled = true;
  if (status) status.textContent = "この位置に記憶中…";
  try {
    const payload = {
      scene_id: sceneId,
      position: {
        x: selected3dPosition.x,
        y: selected3dPosition.y,
        z: selected3dPosition.z,
      },
      kind: "object",
      transcript: "これは" + name + "です",
      detection_confidence:
        selected3dSource?.type === "entity"
          ? 1.0
          : selected3dSource?.type === "component"
            ? 0.95
            : 0.8,
      metadata:
        selected3dSource?.type === "component" && selected3dSource.component
          ? {
              spatial_component_id: selected3dSource.component.id,
              spatial_component_kind: selected3dSource.component.kind,
              spatial_component_source: selected3dSource.component.source,
            }
          : selected3dSource?.type === "shell"
            ? { spatial_model_surface: true }
            : selected3dSource?.type === 'photo-room'
              ? { photo_room_surface: selected3dSource.surfaceId, coordinate_frame: 'vggt_world' }
              : {},
    };
    const response = await fetch("/observations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const entity = await safeJson(response, "記憶に失敗しました");
    if (status) status.innerHTML =
      '<span class="ok">記憶しました: ' + entity.name + "</span>";
    document.getElementById("answer").textContent = "記憶しました: " + entity.name;
    await refresh(entity.id);
    showSelectionMarker(vec(entity.position));
    return entity;
  } finally {
    button.disabled = false;
  }
}

const geometryVariantSelect = document.getElementById("geometryVariant");
let captureViewIndex = 0;
document.getElementById("captureViewBtn")?.addEventListener("click", () => {
  const views = spatialModelManifest?.appearance_2dgs?.capture_views || [];
  if (!views.length) return;
  const view = views[captureViewIndex++ % views.length];
  const distance = Math.max((spatialModelManifest?.metrics?.scene_diagonal || 1) * 0.25, 0.01);
  camera.position.fromArray(view.position);
  camera.up.fromArray(view.up);
  if (Number.isFinite(view.fov_y_degrees)) {
    // Match the capture's vertical field of view; aspect only changes side crop.
    camera.fov = view.fov_y_degrees;
    camera.updateProjectionMatrix();
  }
  // OrbitControls caches the camera-up basis at construction.
  controls.dispose();
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enabled = !walker.active;
  controls.target.copy(camera.position).addScaledVector(new THREE.Vector3().fromArray(view.forward), distance);
  camera.lookAt(controls.target);
  controls.update();
  if (walker.active) walker.start(spatialModelManifest?.metrics?.scene_diagonal || 1);
});
if (geometryVariantSelect) {
  geometryVariantSelect.addEventListener("change", async event => {
    const next = event.target.value;
    if (!["raw", "hybrid", "texture", "2dgs", "fusion", "photo-room"].includes(next)) return;
    const wasPhoto=geometryVariant==='photo-room';
    geometryVariant = next;
    if (["texture", "2dgs", "fusion", "photo-room"].includes(next)) {
      viewDisplayState.pointCloud = false;
      viewDisplayState.mesh = true;
      viewDisplayState.cad = false;
      updateViewDisplay();
    }
    geometryVariantSelect.disabled = true;
    if (walker.active) setWalking(false);
    document.getElementById('walkBtn').disabled = true;
    document.getElementById('appearanceVariant').disabled = true;
    const position = camera.position.clone();
    const target = controls.target.clone();
    const pending = [pointCloudLoadPromise, spatialModelLoadPromise];
    if(wasPhoto){
      photoRoom.clear();walker.constrainMove=null;photoCameraReady=false;
      document.body.classList.remove('has-photo-room');
      camera.up.set(0,1,0);controls.dispose();controls=new OrbitControls(camera,renderer.domElement);controls.target.copy(target);
    }
    disposePointCloud(true);
    disposeSpatialModel();
    try {
      await Promise.allSettled(pending);
      await Promise.all([loadPointCloud(), loadSpatialModel()]);
      if(next!=='photo-room'&&!wasPhoto){camera.position.copy(position);controls.target.copy(target);controls.update();}
      if(next==='photo-room')return;
      document.getElementById("geometryStatus").textContent =
        !spatialModelLoaded || (viewDisplayState.pointCloud && !pointCloudLoaded)
          ? "一部の形状を読み込めませんでした。元の形状に戻して再試行してください。"
          : next === "fusion" ? "複数の撮影画像で奥行きが一致した形状だけを2DGSと重ねています。裏付けのない面は表示しません。"
          : next === "2dgs" ? (appearanceVariant === 'baseline' ? "初回の2DGSです。改善後と同じ視点で比較できます。" : "見た目は2DGS、位置選択はVGGTの形状を使います。同じ座標で表示しています。")
          : next === "texture" ? `写真の適用範囲：約${Math.round((spatialModelManifest?.texture?.textured_area_ratio || 0) * 100)}%。奥行きが一致する構造面に投影し、他の面は元の色を残します。`
          : next === "hybrid" ? "観測された構造面のみ補正。大きな欠損は保持します。" : "元の点群・メッシュを表示中";
    } finally {
      geometryVariantSelect.disabled = false;
      document.getElementById('walkBtn').disabled = false;
      document.getElementById('appearanceVariant').disabled = !spatialModelManifest?.appearance_2dgs_baseline;
      if(next!=='photo-room')document.getElementById('photoRoomState').textContent='元の3Dデータを表示中';
      updateViewDisplay();setWorkspaceScreen(screenFromHash());
    }
  });
}

document.getElementById('appearanceVariant')?.addEventListener('change', event => {
  if (!['current', 'baseline'].includes(event.target.value)) return;
  appearanceVariant = event.target.value;
  if (!['2dgs', 'fusion'].includes(geometryVariant)) geometryVariantSelect.value = '2dgs';
  geometryVariantSelect.dispatchEvent(new Event('change'));
});
document.getElementById("showGrid")?.addEventListener("change", event => {
  if (gridHelper) gridHelper.visible = event.target.checked;
});

const togglePointCloudBtn = document.getElementById("togglePointCloudBtn");
if (togglePointCloudBtn) {
  togglePointCloudBtn.addEventListener("click", () => {
    viewDisplayState.pointCloud = !viewDisplayState.pointCloud;
    if (viewDisplayState.pointCloud) schedulePointCloudLoad();
    else disposePointCloud(true);
    updateViewDisplay();
  });
}
function setPointCloudQuality(quality) {
  if (!Object.hasOwn(pointCloudBudgets, quality) || quality === pointCloudQuality) return;
  pointCloudQuality = quality;
  // Invalidate in-flight loads as well as release the current GPU buffers.
  disposePointCloud(true);
  updateViewDisplay();
  schedulePointCloudLoad();
}

const pointCloudQualitySelect = document.getElementById("pointCloudQuality");
if (pointCloudQualitySelect) {
  pointCloudQualitySelect.value = pointCloudQuality;
  pointCloudQualitySelect.addEventListener("change", event => {
    setPointCloudQuality(event.target.value);
  });
}

const toggleMeshBtn = document.getElementById("toggleMeshBtn");
if (toggleMeshBtn) {
  toggleMeshBtn.addEventListener("click", () => {
    viewDisplayState.mesh = !viewDisplayState.mesh;
    updateViewDisplay();
  });
}
const toggleCadBtn = document.getElementById("toggleCadBtn");
if (toggleCadBtn) {
  toggleCadBtn.addEventListener("click", () => {
    viewDisplayState.cad = !viewDisplayState.cad;
    updateViewDisplay();
  });
}
const pointSizeSlider = document.getElementById("pointSizeSlider");
if (pointSizeSlider) {
  pointSizeSlider.addEventListener("input", event => {
    viewDisplayState.pointSize = Number(event.target.value);
    updateViewDisplay();
  });
}

document.getElementById("select3dBtn").addEventListener("click", () => {
  if (walker.active) setWalking(false);
  set3dSelectMode(!select3dMode);
});
document.getElementById("cancel3dSelectionBtn").addEventListener("click", () => {
  clear3dSelection({ keepMode: true });
  const status = document.getElementById("selectionStatus");
  if (status) status.textContent = "3D空間の物をタップしてください";
});
document.getElementById("save3dSelectionBtn").addEventListener("click", () => {
  save3dSelection().catch(error => {
    const status = document.getElementById("selectionStatus");
    if (status) status.innerHTML = '<span class="warn">' + error.message + "</span>";
  });
});
renderer.domElement.addEventListener("pointerdown", event => {
  if (!select3dMode || event.button > 0) return;
  selectionPointerDown = { x: event.clientX, y: event.clientY };
});
renderer.domElement.addEventListener("pointerup", event => {
  if (!select3dMode || !selectionPointerDown) return;
  const distance = Math.hypot(
    event.clientX - selectionPointerDown.x,
    event.clientY - selectionPointerDown.y
  );
  selectionPointerDown = null;
  if (distance <= 8) select3dAtPointer(event.clientX, event.clientY);
});
renderer.domElement.addEventListener("pointercancel", () => {
  selectionPointerDown = null;
});

async function refresh(focusId = null) {
  let url = "/viewer/state/" + encodeURIComponent(sceneId);
  if (focusId) url += "?focus=" + encodeURIComponent(focusId);
  const response = await fetch(url);
  const state = await safeJson(response, "Viewer状態の取得に失敗しました");
  document.getElementById("sceneLabel").textContent = state.scene.name || "保存済みの部屋";
  await choosePhotoRoom();
  const nextPointCloudRevision = state.scene.point_cloud_path || "";
  const nextSpatialModelPath = state.scene.spatial_model_path || "";

  if (nextPointCloudRevision !== pointCloudRevision) {
    pointCloudRevision = nextPointCloudRevision;
    disposePointCloud();
  }
  if (nextSpatialModelPath !== spatialModelPath) {
    spatialModelPath = nextSpatialModelPath;
    disposeSpatialModel();
  }

  renderEntities(state.entities, focusId);

  const loadTasks = [];
  if (spatialModelPath || geometryVariant==='photo-room') {
    if (mobileViewport && geometryVariant!=='photo-room') scheduleSpatialModelLoad();
    else loadTasks.push(loadSpatialModel());
  }
  if (state.scene.point_cloud_path) {
    if (mobileViewport) schedulePointCloudLoad();
    else loadTasks.push(loadPointCloud());
  }
  if (loadTasks.length) {
    await Promise.allSettled(loadTasks);
  }
  updateViewDisplay();
  return state;
}

async function guide() {
  const start = {
    x: Number(document.getElementById("sx").value),
    y: Number(document.getElementById("sy").value),
    z: Number(document.getElementById("sz").value),
  };
  const text = document.getElementById("query").value.trim();
  if (!text) throw new Error("検索文を入力してください");

  const response = await fetch("/guide", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_id: sceneId, start: start, text: text }),
  });
  const data = await safeJson(response, "ルート案内に失敗しました");

  const target = data.query.target;
  const answer = document.getElementById("answer");
  answer.innerHTML = "<span class=\"ok\">" + target.name + "</span><br>" +
    "最後に確認: (" + target.position.x.toFixed(2) + ", " + target.position.y.toFixed(2) + ", " + target.position.z.toFixed(2) + ")<br>" +
    "confidence: " + target.confidence.toFixed(2);
  await refresh(target.id);
  renderRoute(data.route.points);
  focusMemoryTarget(target);
}

async function addMemory() {
  const payload = {
    scene_id: sceneId,
    position: {
      x: Number(document.getElementById("mx").value),
      y: Number(document.getElementById("my").value),
      z: Number(document.getElementById("mz").value),
    },
    kind: "object",
    transcript: document.getElementById("memo").value,
    detected_label: document.getElementById("label").value || null,
    detection_confidence: 0.9,
  };
  const response = await fetch("/observations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await safeJson(response, "記憶の追加に失敗しました");
  document.getElementById("answer").textContent = "記憶しました: " + data.name;
  await refresh(data.id);
}
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;
let mediaStream = null;

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("このブラウザではマイク録音を利用できません");
  }
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recordedChunks = [];
  recordedBlob = null;
  mediaRecorder = new MediaRecorder(mediaStream);
  mediaRecorder.ondataavailable = event => {
    if (event.data && event.data.size > 0) recordedChunks.push(event.data);
  };
  mediaRecorder.onstop = () => {
    const mime = mediaRecorder && mediaRecorder.mimeType ? mediaRecorder.mimeType : "audio/webm";
    recordedBlob = new Blob(recordedChunks, { type: mime });
    document.getElementById("captureStatus").innerHTML =
      "<span class=\"ok\">録音済み</span> " + Math.round(recordedBlob.size / 1024) + " KB";
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }
  };
  mediaRecorder.start();
  document.getElementById("recordBtn").disabled = true;
  document.getElementById("stopBtn").disabled = false;
  document.getElementById("captureStatus").textContent = "録音中…";
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") return;
  mediaRecorder.stop();
  document.getElementById("recordBtn").disabled = false;
  document.getElementById("stopBtn").disabled = true;
}

function currentAudioFile() {
  const uploaded = document.getElementById("captureAudio").files[0];
  if (uploaded) return uploaded;
  if (recordedBlob) {
    const ext = recordedBlob.type.includes("ogg") ? "ogg" : "webm";
    return new File([recordedBlob], "recording." + ext, { type: recordedBlob.type || "audio/webm" });
  }
  return null;
}
async function captureMemory() {
  const imageFile = await getCaptureImageFile();
  const audioFile = currentAudioFile();
  if (!imageFile) throw new Error("対象が写っている画像またはSceneフレームが必要です");
  if (!audioFile) throw new Error("録音するか音声ファイルを選択してください");

  const status = document.getElementById("captureStatus");
  const button = document.getElementById("captureBtn");
  status.textContent = "AIが記憶を作成中…";
  button.disabled = true;

  const form = new FormData();
  form.append("image", imageFile);
  form.append("audio", audioFile);
  form.append("frame_index", document.getElementById("captureFrame").value || "0");
  form.append("language", "ja");
  const explicitPrompt = document.getElementById("capturePrompt").value.trim();
  if (explicitPrompt) form.append("prompt", explicitPrompt);

  try {
    const response = await fetch("/scenes/" + encodeURIComponent(sceneId) + "/voice/remember", {
      method: "POST",
      body: form,
    });
    const data = await safeJson(response, "AI記憶の作成に失敗しました");

    const entity = data.entity;
    const transcript = data.transcription.text;
    status.innerHTML =
      "<span class=\"ok\">記憶しました: " + entity.name + "</span><br>" +
      "音声: " + transcript + "<br>" +
      "検出: " + data.detection.label + " (" + data.detection.score.toFixed(2) + ")<br>" +
      "3D: (" + entity.position.x.toFixed(2) + ", " + entity.position.y.toFixed(2) + ", " + entity.position.z.toFixed(2) + ")";
    document.getElementById("answer").textContent = "記憶しました: " + entity.name;
    await refresh(entity.id);
    controls.target.copy(vec(entity.position));
    controls.update();
  } finally {
    button.disabled = false;
  }
}

function setupBrowserSpeechSearch() {
  document.getElementById("voiceBtn").onclick = () => {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      alert("このブラウザはWeb Speech APIに対応していません");
      return;
    }
    const recognition = new Recognition();
    recognition.lang = "ja-JP";
    recognition.onresult = event => {
      document.getElementById("query").value = event.results[0][0].transcript;
    };
    recognition.start();
  };
}
document.getElementById("guideBtn").onclick = () => guide().catch(e => alert(e.message));
document.getElementById("memoryBtn").onclick = () => addMemory().catch(e => alert(e.message));
document.getElementById("recordBtn").onclick = () => startRecording().catch(e => alert(e.message));
document.getElementById("stopBtn").onclick = () => stopRecording();
document.getElementById("captureBtn").onclick = () => captureMemory().catch(e => {
  document.getElementById("captureStatus").innerHTML = "<span class=\"warn\">" + e.message + "</span>";
});

setupBrowserSpeechSearch();
refresh().catch(error => {
  console.error(error);
  document.getElementById("answer").textContent = "Scene load failed: " + error.message;
});
let sceneFrameManifest = null;

let frameDetections = [];
let selectedFrameDetection = -1;

function updateFrameIndicator() {
  const indicator = document.getElementById("frameIndicator");
  if (!indicator) return;
  const count = sceneFrameManifest?.count || 0;
  const index = Number(document.getElementById("captureFrame")?.value || 0);
  indicator.textContent = count ? "frame " + (index + 1) + " / " + count : "frame 0 / 0";
}

function clearFrameCandidates(message = null) {
  frameDetections = [];
  selectedFrameDetection = -1;
  const saveButton = document.getElementById("saveLabelBtn");
  if (saveButton) saveButton.disabled = true;
  drawFrameCandidates();
  if (message) {
    const status = document.getElementById("labelStatus");
    if (status) status.textContent = message;
  }
}

async function loadAudioMemoryReport() {
  const status = document.getElementById("audioMemoryStatus");
  if (!status) return null;

  let data = null;
  try {
    const response = await fetch(
      "/scenes/" + encodeURIComponent(sceneId) + "/audio-memory"
    );
    if (response.ok) {
      data = await response.json();
    } else if (response.status !== 404) {
      throw new Error(await response.text());
    }
  } catch (error) {
    console.warn("Audio memory report unavailable:", error);
  }

  if (!data) {
    try {
      const cached = sessionStorage.getItem(
        "spatial:last-audio-memory:" + sceneId
      );
      if (cached) data = JSON.parse(cached);
    } catch {}
  }

  status.innerHTML = "";
  if (!data || (!data.confirmed_count && !data.pending_count && !data.transcript)) {
    status.textContent = "動画音声から自動登録された記憶はありません。";
    return data;
  }

  const summary = document.createElement("div");
  summary.innerHTML =
    '<span class="ok">自動確定 ' + Number(data.confirmed_count || 0) + "件</span> / " +
    '<span class="' + (Number(data.pending_count || 0) ? "warn" : "muted") + '">' +
    "確認待ち " + Number(data.pending_count || 0) + "件</span>";
  status.appendChild(summary);

  if (data.transcript) {
    const transcript = document.createElement("div");
    transcript.className = "muted";
    transcript.style.marginTop = "6px";
    transcript.textContent = "音声: " + data.transcript;
    status.appendChild(transcript);
  }

  const pending = Array.isArray(data.pending) ? data.pending : [];
  if (pending.length) {
    const hint = document.createElement("div");
    hint.className = "muted";
    hint.style.marginTop = "8px";
    hint.textContent = "確認待ちはフレームを開いて名前を確認できます。";
    status.appendChild(hint);

    for (const item of pending.slice(0, 8)) {
      const button = document.createElement("button");
      button.className = "secondary";
      button.style.textAlign = "left";
      button.textContent =
        "frame " + (Number(item.frame_index || 0) + 1) +
        "｜" + (item.text || "発話") +
        "｜" + (item.reason || "確認待ち");
      button.addEventListener("click", () => {
        const slider = document.getElementById("captureFrame");
        slider.value = String(item.frame_index || 0);
        slider.dispatchEvent(new Event("input", { bubbles: true }));
        document.getElementById("captureName").value =
          String(item.text || "")
            .replace(/^これは/, "")
            .replace(/(?:です|だよ|だ)[。！!？?]*$/, "")
            .trim();
      });
      status.appendChild(button);
    }
  }
  return data;
}

async function loadFrameManifest() {
  try {
    const response = await fetch("/scenes/" + encodeURIComponent(sceneId) + "/frames");
    if (!response.ok) return null;
    sceneFrameManifest = await response.json();
    const input = document.getElementById("captureFrame");
    input.max = String(Math.max((sceneFrameManifest.count || 1) - 1, 0));
    input.value = String(Math.min(Number(input.value || 0), Number(input.max || 0)));
    updateFrameIndicator();
    return sceneFrameManifest;
  } catch (error) {
    console.warn("Frame manifest unavailable:", error);
    return null;
  }
}

async function previewSceneFrame() {
  if (!sceneFrameManifest) await loadFrameManifest();
  if (!sceneFrameManifest || !sceneFrameManifest.count) {
    throw new Error("このSceneには抽出済みフレームがありません");
  }
  const index = Number(document.getElementById("captureFrame").value || 0);
  if (index < 0 || index >= sceneFrameManifest.count) throw new Error("frame indexが範囲外です");
  const preview = document.getElementById("framePreview");
  const nextSrc = "/scenes/" + encodeURIComponent(sceneId) + "/frames/" + index;
  if (!preview.src.endsWith(nextSrc)) {
    await new Promise((resolve, reject) => {
      preview.onload = () => resolve();
      preview.onerror = () => reject(new Error("Sceneフレームを読み込めません"));
      preview.src = nextSrc;
    });
  } else if (!preview.complete) {
    await new Promise(resolve => preview.addEventListener("load", resolve, { once: true }));
  }
  updateFrameIndicator();
  drawFrameCandidates();
}

async function getCaptureImageFile() {
  const uploaded = document.getElementById("captureImage").files[0];
  if (uploaded) return uploaded;
  return getSelectedSceneFrameFile();
}

async function getSelectedSceneFrameFile() {
  if (!sceneFrameManifest) await loadFrameManifest();
  if (!sceneFrameManifest || !sceneFrameManifest.count) return null;
  const index = Number(document.getElementById("captureFrame").value || 0);
  if (index < 0 || index >= sceneFrameManifest.count) return null;
  const response = await fetch("/scenes/" + encodeURIComponent(sceneId) + "/frames/" + index);
  if (!response.ok) return null;
  const blob = await response.blob();
  return new File([blob], "scene-frame-" + index + ".jpg", { type: blob.type || "image/jpeg" });
}

function frameDisplayTransform(detection) {
  const preview = document.getElementById("framePreview");
  const canvas = document.getElementById("frameOverlay");
  const width = Math.max(preview.clientWidth, 1);
  const height = Math.max(preview.clientHeight, 1);
  canvas.width = width;
  canvas.height = height;
  const scale = Math.min(
    width / Math.max(detection.image_width, 1),
    height / Math.max(detection.image_height, 1)
  );
  return {
    scale,
    offsetX: (width - detection.image_width * scale) / 2,
    offsetY: (height - detection.image_height * scale) / 2,
  };
}

function drawFrameCandidates() {
  const canvas = document.getElementById("frameOverlay");
  const preview = document.getElementById("framePreview");
  if (!canvas || !preview || !preview.clientWidth || !preview.clientHeight) return;
  canvas.width = Math.max(preview.clientWidth, 1);
  canvas.height = Math.max(preview.clientHeight, 1);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  frameDetections.forEach((detection, index) => {
    const transform = frameDisplayTransform(detection);
    const [x1, y1, x2, y2] = detection.bbox_xyxy;
    const x = transform.offsetX + x1 * transform.scale;
    const y = transform.offsetY + y1 * transform.scale;
    const w = (x2 - x1) * transform.scale;
    const h = (y2 - y1) * transform.scale;
    const selected = index === selectedFrameDetection;
    ctx.lineWidth = selected ? 5 : 3;
    ctx.strokeStyle = selected ? "#48e07a" : "#ffcc00";
    ctx.strokeRect(x, y, w, h);
    const label = detection.label + " " + Number(detection.score || 0).toFixed(2);
    ctx.font = "bold 14px system-ui";
    const tw = ctx.measureText(label).width + 10;
    ctx.fillStyle = "rgba(0,0,0,.76)";
    ctx.fillRect(x, Math.max(0, y - 22), tw, 21);
    ctx.fillStyle = selected ? "#48e07a" : "#ffcc00";
    ctx.fillText(label, x + 5, Math.max(15, y - 6));
  });
}

function selectFrameCandidateFromEvent(event) {
  if (!frameDetections.length) return;
  const canvas = document.getElementById("frameOverlay");
  const rect = canvas.getBoundingClientRect();
  const px = (event.clientX - rect.left) * (canvas.width / Math.max(rect.width, 1));
  const py = (event.clientY - rect.top) * (canvas.height / Math.max(rect.height, 1));
  let best = -1;
  let bestDistance = Infinity;
  frameDetections.forEach((detection, index) => {
    const transform = frameDisplayTransform(detection);
    const [x1, y1, x2, y2] = detection.bbox_xyxy;
    const left = transform.offsetX + x1 * transform.scale;
    const top = transform.offsetY + y1 * transform.scale;
    const right = transform.offsetX + x2 * transform.scale;
    const bottom = transform.offsetY + y2 * transform.scale;
    if (px < left || px > right || py < top || py > bottom) return;
    const cx = (left + right) / 2;
    const cy = (top + bottom) / 2;
    const distance = Math.hypot(px - cx, py - cy);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  if (best >= 0) {
    selectedFrameDetection = best;
    document.getElementById("saveLabelBtn").disabled = false;
    const detection = frameDetections[best];
    document.getElementById("labelStatus").innerHTML =
      '<span class="ok">選択中: ' + detection.label + " (" +
      Number(detection.score || 0).toFixed(2) + ")</span><br>この枠を記憶できます";
    drawFrameCandidates();
  }
}

async function detectFrameCandidates() {
  const name = document.getElementById("captureName").value.trim();
  if (!name) throw new Error("この物の名前を入力してください");
  await previewSceneFrame();
  const image = await getSelectedSceneFrameFile();
  if (!image) throw new Error("Sceneフレームを取得できません");

  const button = document.getElementById("detectFrameBtn");
  const status = document.getElementById("labelStatus");
  button.disabled = true;
  status.textContent = "AIがこのフレームから候補を探しています…";
  clearFrameCandidates();
  const form = new FormData();
  form.append("image", image);
  form.append("transcript", name);
  const explicitPrompt = document.getElementById("capturePrompt").value.trim();
  if (explicitPrompt) form.append("prompt", explicitPrompt);
  try {
    const response = await fetch("/vision/detect", { method: "POST", body: form });
    const data = await safeJson(response, "対象候補を検出できません");
    frameDetections = data.detections || [];
    if (!frameDetections.length) throw new Error("候補が見つかりませんでした");

    selectedFrameDetection = 0;
    if (data.target) {
      const targetBox = JSON.stringify(data.target.bbox_xyxy);
      const targetIndex = frameDetections.findIndex(
        detection => JSON.stringify(detection.bbox_xyxy) === targetBox
      );
      if (targetIndex >= 0) selectedFrameDetection = targetIndex;
    }
    document.getElementById("saveLabelBtn").disabled = false;
    const selected = frameDetections[selectedFrameDetection];
    status.innerHTML =
      '<span class="ok">' + frameDetections.length + "件の候補</span><br>" +
      "AI選択: " + selected.label + " (" + Number(selected.score || 0).toFixed(2) + ")" +
      (frameDetections.length > 1 ? "<br>別の枠をタップすると変更できます" : "");
    drawFrameCandidates();
  } finally {
    button.disabled = false;
  }
}

async function saveSelectedFrameLabel() {
  const name = document.getElementById("captureName").value.trim();
  if (!name) throw new Error("この物の名前を入力してください");
  const detection = frameDetections[selectedFrameDetection];
  if (!detection) throw new Error("先に対象候補を選択してください");
  const frameIndex = Number(document.getElementById("captureFrame").value || 0);
  const button = document.getElementById("saveLabelBtn");
  const status = document.getElementById("labelStatus");
  button.disabled = true;
  status.textContent = "3D位置を計算して記憶中…";

  const response = await fetch(
    "/scenes/" + encodeURIComponent(sceneId) + "/detections/ingest",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frame_index: frameIndex,
        bbox_xyxy: detection.bbox_xyxy.map(value => Math.round(value)),
        source_width: detection.image_width,
        source_height: detection.image_height,
        transcript: "これは" + name + "です",
        detected_label: detection.label,
        detection_confidence: detection.score,
        kind: "object",
      }),
    }
  );
  const entity = await safeJson(response, "記憶に失敗しました");
  status.innerHTML =
    '<span class="ok">記憶しました: ' + entity.name + "</span><br>" +
    "3D: (" + entity.position.x.toFixed(2) + ", " +
    entity.position.y.toFixed(2) + ", " + entity.position.z.toFixed(2) + ")";
  document.getElementById("answer").textContent = "記憶しました: " + entity.name;
  await refresh(entity.id);
  button.disabled = false;
}

async function reconstructScene() {
  const video = document.getElementById("sceneVideo").files[0];
  if (!video) throw new Error("動画ファイルを選択してください");
  const status = document.getElementById("reconstructStatus");
  const button = document.getElementById("reconstructBtn");
  const form = new FormData();
  form.append("video", video);
  form.append("name", document.getElementById("sceneName").value.trim() || "New Scene");
  form.append("sample_fps", document.getElementById("sampleFps").value || "2");
  form.append("max_frames", document.getElementById("maxFrames").value || "60");
  form.append("point_stride", "4");

  button.disabled = true;
  status.textContent = "3Dシーンを生成中…";
  try {
    const response = await fetch("/scenes/reconstruct", { method: "POST", body: form });
    const data = await safeJson(response, "3Dシーン再構成に失敗しました");
    status.innerHTML = "<span class=\"ok\">生成完了</span><br>" +
      "model: " + (data.scene.reconstruction_model || data.metrics.model || "unknown") + "<br>" +
      "frames: " + data.scene.frame_count + "<br>" +
      "VRAM peak: " + Number(data.metrics.inference_peak_allocated_gib || 0).toFixed(2) + " GiB";
    window.location.href = "/?scene=" + encodeURIComponent(data.scene.id);
  } finally {
    button.disabled = false;
  }
}

document.getElementById("previewFrameBtn").onclick = () => previewSceneFrame().catch(e => alert(e.message));
document.getElementById("reconstructBtn").onclick = () => reconstructScene().catch(e => {
  document.getElementById("reconstructStatus").innerHTML = "<span class=\"warn\">" + e.message + "</span>";
});

let framePreviewTimer = null;
const captureFrameInput = document.getElementById("captureFrame");
captureFrameInput.addEventListener("input", () => {
  updateFrameIndicator();
  clearFrameCandidates("フレームを変更しました。名前を入力して候補を探してください");
  if (framePreviewTimer) clearTimeout(framePreviewTimer);
  framePreviewTimer = window.setTimeout(() => {
    previewSceneFrame().catch(error => {
      document.getElementById("labelStatus").innerHTML =
        '<span class="warn">' + error.message + "</span>";
    });
  }, 80);
});
document.getElementById("prevFrameBtn").onclick = () => {
  captureFrameInput.value = String(Math.max(
    Number(captureFrameInput.min || 0),
    Number(captureFrameInput.value || 0) - 1
  ));
  captureFrameInput.dispatchEvent(new Event("input", { bubbles: true }));
};
document.getElementById("nextFrameBtn").onclick = () => {
  captureFrameInput.value = String(Math.min(
    Number(captureFrameInput.max || 0),
    Number(captureFrameInput.value || 0) + 1
  ));
  captureFrameInput.dispatchEvent(new Event("input", { bubbles: true }));
};
document.getElementById("captureName").addEventListener("input", () => {
  if (frameDetections.length) clearFrameCandidates("名前を変更しました。候補を探し直してください");
});
document.getElementById("capturePrompt").addEventListener("input", () => {
  if (frameDetections.length) clearFrameCandidates("検出設定を変更しました。候補を探し直してください");
});
document.getElementById("frameOverlay").addEventListener("click", selectFrameCandidateFromEvent);
document.getElementById("detectFrameBtn").onclick = () => detectFrameCandidates().catch(error => {
  document.getElementById("labelStatus").innerHTML =
    '<span class="warn">' + error.message + "</span>";
});
document.getElementById("saveLabelBtn").onclick = () => saveSelectedFrameLabel().catch(error => {
  document.getElementById("labelStatus").innerHTML =
    '<span class="warn">' + error.message + "</span>";
  document.getElementById("saveLabelBtn").disabled = selectedFrameDetection < 0;
});
window.addEventListener("resize", drawFrameCandidates);
loadFrameManifest().then(() => {
  if (screenFromHash() === "memory") previewSceneFrame().catch(() => {});
});
loadAudioMemoryReport().catch(() => {});
async function autoIndexScene() {
  const status = document.getElementById("autoIndexStatus");
  const button = document.getElementById("autoIndexBtn");
  const form = new FormData();
  form.append("frame_step", "5");
  form.append("max_frames", "12");
  form.append("max_detections_per_frame", "20");
  form.append("threshold", "0.35");
  form.append("text_threshold", "0.25");
  form.append("cluster_radius", "0.45");
  button.disabled = true;
  status.textContent = "AIがSceneを巡回して記憶中…";
  try {
    const response = await fetch("/scenes/" + encodeURIComponent(sceneId) + "/auto-index", {
      method: "POST",
      body: form,
    });
    const data = await safeJson(response, "自動記憶に失敗しました");
    status.innerHTML = "<span class=\"ok\">自動記憶完了</span><br>" +
      "frames: " + data.frames_processed + "<br>" +
      "detections: " + data.detections_seen +
      " → clusters: " + data.clusters_built + "<br>" +
      "multi-view clusters: " + data.multi_observation_clusters + "<br>" +
      "entities updated: " + data.entity_count;
    await refresh();
  } finally {
    button.disabled = false;
  }
}

document.getElementById("autoIndexBtn").onclick = () => autoIndexScene().catch(e => {
  document.getElementById("autoIndexStatus").innerHTML = "<span class=\"warn\">" + e.message + "</span>";
});
async function askAI() {
  const start = {
    x: Number(document.getElementById("sx").value),
    y: Number(document.getElementById("sy").value),
    z: Number(document.getElementById("sz").value),
  };
  const text = document.getElementById("query").value.trim();
  if (!text) throw new Error("質問を入力してください");

  const response = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_id: sceneId, text: text, start: start, limit: 5 }),
  });
  const data = await safeJson(response, "AIへの質問に失敗しました");

  const answer = document.getElementById("answer");
  answer.innerHTML = "<span class=\"ok\">" + data.answer + "</span><br>" +
    "<span class=\"muted\">source: " + data.answer_source + "</span>";

  const target = data.query && data.query.target ? data.query.target : null;
  if (target) {
    await refresh(target.id);
    focusMemoryTarget(target);
  }
  if (data.route && data.route.points) renderRoute(data.route.points);
}

document.getElementById("askBtn").onclick = () => askAI().catch(e => alert(e.message));
let cameraStream = null;
let cameraState = "idle";
let cameraStoppingInternally = false;

function cameraStreamIsLive() {
  return !!cameraStream &&
    cameraStream.getVideoTracks().some(track => track.readyState === "live");
}

function setCameraButtonState(state) {
  cameraState = state;
  const button = document.getElementById("cameraStartBtn");
  if (!button) return;
  const labels = {
    idle: "カメラ開始",
    starting: "カメラ起動中…",
    active: "カメラ停止",
    stopping: "カメラ停止中…",
    error: "カメラ再試行",
  };
  button.dataset.cameraState = state;
  button.textContent = labels[state] || labels.idle;
  button.disabled = state === "starting" || state === "stopping";
  button.classList.toggle("camera-active", state === "active");
  button.setAttribute("aria-pressed", state === "active" ? "true" : "false");
}

function stopCameraStream({ quiet = true, nextState = "idle" } = {}) {
  stopContinuousTracking(true);
  cameraStoppingInternally = true;
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  cameraStoppingInternally = false;
  const video = document.getElementById("cameraVideo");
  if (video) {
    try { video.pause(); } catch {}
    video.srcObject = null;
  }
  clearCameraOverlay();
  setCameraButtonState(nextState);
  if (!quiet) {
    document.getElementById("cameraStatus").textContent = "カメラ停止";
  }
}

async function startCamera() {
  if (cameraState === "starting" || cameraState === "stopping") return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setCameraButtonState("error");
    throw new Error("このブラウザではカメラを利用できません");
  }

  if (cameraStream) stopCameraStream();
  setCameraButtonState("starting");
  const status = document.getElementById("cameraStatus");
  status.textContent = "カメラを起動中…";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: mobileViewport ? 1280 : 1920 },
        height: { ideal: mobileViewport ? 720 : 1080 },
      },
      audio: false,
    });
    cameraStream = stream;
    for (const track of stream.getVideoTracks()) {
      track.addEventListener("ended", () => {
        if (!cameraStoppingInternally && cameraStream === stream) {
          cameraStream = null;
          stopContinuousTracking(true);
          setCameraButtonState("idle");
          const currentStatus = document.getElementById("cameraStatus");
          if (currentStatus) currentStatus.textContent = "カメラが停止しました";
        }
      });
    }

    const video = document.getElementById("cameraVideo");
    video.srcObject = stream;
    await video.play();
    if (!cameraStreamIsLive()) throw new Error("カメラ映像を開始できませんでした");
    setCameraButtonState("active");
    status.textContent = "カメラ起動。3D位置合わせを準備中…";
    clearCameraOverlay();

    try {
      const prepared = await warmRelocalization();
      const cacheSource = prepared.index && prepared.index.source
        ? prepared.index.source
        : "unknown";
      if (cameraStreamIsLive()) {
        status.innerHTML = "<span class=\"ok\">カメラ準備完了</span><br>" +
          "reference frames: " + prepared.reference_frames +
          " / index: " + cacheSource +
          " / " + prepared.seconds.toFixed(2) + "s";
      }
    } catch (error) {
      if (cameraStreamIsLive()) {
        status.innerHTML = "<span class=\"warn\">カメラ起動済み / 位置合わせ準備失敗</span><br>" +
          error.message;
      }
    }
  } catch (error) {
    stopCameraStream({ quiet: true, nextState: "error" });
    status.innerHTML = "<span class=\"warn\">カメラ起動失敗</span><br>" + error.message;
    throw error;
  }
}

async function stopCamera() {
  if (cameraState === "starting" || cameraState === "stopping") return;
  setCameraButtonState("stopping");
  await new Promise(resolve => requestAnimationFrame(resolve));
  stopCameraStream({ quiet: false, nextState: "idle" });
}

async function toggleCamera() {
  if (cameraState === "starting" || cameraState === "stopping") return;
  if (cameraStreamIsLive() || cameraState === "active") {
    await stopCamera();
  } else {
    await startCamera();
  }
}

setCameraButtonState("idle");

function stopAudioResources() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    try { mediaRecorder.stop(); } catch {}
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
}

function clearCameraOverlay() {
  const canvas = document.getElementById("cameraOverlay");
  const ctx = canvas.getContext("2d");
  canvas.width = Math.max(document.getElementById("cameraVideo").clientWidth, 1);
  canvas.height = Math.max(document.getElementById("cameraVideo").clientHeight, 1);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawDetectionBox(detection, label) {
  const video = document.getElementById("cameraVideo");
  const canvas = document.getElementById("cameraOverlay");
  canvas.width = Math.max(video.clientWidth, 1);
  canvas.height = Math.max(video.clientHeight, 1);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const sx = canvas.width / detection.image_width;
  const sy = canvas.height / detection.image_height;
  const box = detection.bbox_xyxy;
  const x = box[0] * sx;
  const y = box[1] * sy;
  const w = (box[2] - box[0]) * sx;
  const h = (box[3] - box[1]) * sy;

  ctx.lineWidth = 4;
  ctx.strokeStyle = "#ffcc00";
  ctx.strokeRect(x, y, w, h);
  ctx.font = "bold 16px system-ui";
  const text = label + " " + detection.score.toFixed(2);
  const tw = ctx.measureText(text).width + 12;
  ctx.fillStyle = "rgba(0,0,0,.75)";
  ctx.fillRect(x, Math.max(0, y - 26), tw, 24);
  ctx.fillStyle = "#ffcc00";
  ctx.fillText(text, x + 6, Math.max(17, y - 8));
}

async function cameraFrameFile() {
  const video = document.getElementById("cameraVideo");
  if (!cameraStream || !video.videoWidth || !video.videoHeight) return null;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  const jpegQuality = mobileViewport ? 0.78 : 0.9;
  const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", jpegQuality));
  if (!blob) return null;
  return new File([blob], "camera.jpg", { type: "image/jpeg" });
}

async function locateInCamera() {
  const query = document.getElementById("query").value.trim();
  if (!query) throw new Error("先に検索対象を入力してください");
  const image = await cameraFrameFile();
  if (!image) throw new Error("カメラを開始してください");

  const form = new FormData();
  form.append("image", image);
  form.append("query", query);
  form.append("verify_visible", "true");
  form.append("threshold", "0.30");
  form.append("text_threshold", "0.25");
  form.append("max_reference_frames", "60");

  const status = document.getElementById("cameraStatus");
  status.textContent = "3D Sceneへ再ローカライズ中…";
  const response = await fetch("/scenes/" + encodeURIComponent(sceneId) + "/ar-guide", {
    method: "POST",
    body: form,
  });
  const data = await safeJson(response, "AR guidance failed");

  lockedTargetId = data.target_entity.id;
  applyARGuidance(data);
}

document.getElementById("cameraStartBtn").onclick = () => toggleCamera().catch(error => {
  console.warn("camera toggle failed", error);
});
document.getElementById("cameraLocateBtn").onclick = () => locateInCamera().catch(e => {
  document.getElementById("cameraStatus").innerHTML = "<span class=\"warn\">" + e.message + "</span>";
});
window.addEventListener("resize", () => { if (cameraStream) clearCameraOverlay(); });

function drawGuidanceArrow(guidance, label) {
  const video = document.getElementById("cameraVideo");
  const canvas = document.getElementById("cameraOverlay");
  canvas.width = Math.max(video.clientWidth, 1);
  canvas.height = Math.max(video.clientHeight, 1);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const dir = guidance.arrow;
  const radius = Math.min(canvas.width, canvas.height) * 0.32;
  const ex = cx + dir.x * radius;
  const ey = cy + dir.y * radius;

  ctx.lineWidth = 7;
  ctx.strokeStyle = "#ffcc00";
  ctx.fillStyle = "#ffcc00";
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(ex, ey);
  ctx.stroke();

  const angle = Math.atan2(ey - cy, ex - cx);
  const head = 22;
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(
    ex - head * Math.cos(angle - Math.PI / 6),
    ey - head * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    ex - head * Math.cos(angle + Math.PI / 6),
    ey - head * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();

  const text = label + "  " + guidance.distance.toFixed(1) + "m";
  ctx.font = "bold 17px system-ui";
  const tw = ctx.measureText(text).width + 18;
  ctx.fillStyle = "rgba(0,0,0,.78)";
  ctx.fillRect(Math.max(4, cx - tw / 2), Math.max(4, cy - 42), tw, 30);
  ctx.fillStyle = "#ffcc00";
  ctx.fillText(text, Math.max(12, cx - tw / 2 + 9), Math.max(25, cy - 19));
}


let lockedTargetId = null;
let continuousTracking = false;
let trackingIteration = 0;
let trackingSessionId = null;

async function warmRelocalization() {
  const form = new FormData();
  form.append("max_reference_frames", "60");
  const response = await fetch(
    "/scenes/" + encodeURIComponent(sceneId) + "/relocalization/prepare",
    { method: "POST", body: form }
  );
  const data = await safeJson(response, "relocalization prepare failed");
  return data;
}

function applyARGuidance(data) {
  const status = document.getElementById("cameraStatus");
  const poseConfidence = Number(data.pose?.confidence || 0);
  if (data.pose && data.pose.camera_position_world) {
    const p = data.pose.camera_position_world;
    document.getElementById("sx").value = Number(p.x).toFixed(3);
    document.getElementById("sy").value = Number(p.y).toFixed(3);
    document.getElementById("sz").value = Number(p.z).toFixed(3);
  }

  if (poseConfidence < 0.60) {
    clearCameraOverlay();
    status.innerHTML =
      "<span class=\"warn\">位置合わせ中…</span><br>" +
      "pose confidence: " + poseConfidence.toFixed(2) +
      " / inliers: " + Number(data.pose?.inliers || 0) +
      " / refs: " + Number(data.pose?.references_tested || 0);
    return;
  }

  if (data.mode === "bbox" && data.detection) {
    drawDetectionBox(data.detection, data.target_entity.name);
    status.innerHTML = "<span class=\"ok\">実物を確認: " +
      data.target_entity.name + "</span><br>" +
      "vision: " + data.detection.score.toFixed(2) +
      " / 距離: " + data.guidance.distance.toFixed(2) + "<br>" +
      "pose: " + data.pose.inliers + " inliers / " +
      data.pose.reprojection_error.toFixed(2) + "px / conf " +
      Number(data.pose.confidence || 0).toFixed(2) +
      " / refs " + (data.pose.references_tested || "-");
  } else {
    drawGuidanceArrow(data.guidance, data.target_entity.name);
    status.innerHTML = "<span class=\"ok\">追跡中: " +
      data.target_entity.name + "</span><br>" +
      "距離: " + data.guidance.distance.toFixed(2) + "<br>" +
      "pose: " + data.pose.inliers + " inliers / " +
      data.pose.reprojection_error.toFixed(2) + "px / conf " +
      Number(data.pose.confidence || 0).toFixed(2) +
      " / refs " + (data.pose.references_tested || "-");
  }
}


async function trackLockedTargetOnce() {
  if (!lockedTargetId) {
    await locateInCamera();
    if (!lockedTargetId) return;
  }

  const image = await cameraFrameFile();
  if (!image) throw new Error("カメラを開始してください");

  const form = new FormData();
  form.append("image", image);
  form.append("target_entity_id", lockedTargetId);
  form.append("tracking_session", trackingSessionId || "default");
  form.append("verify_visible", trackingIteration % 3 === 0 ? "true" : "false");
  form.append("threshold", "0.30");
  form.append("text_threshold", "0.25");
  form.append("max_reference_frames", "60");

  const response = await fetch(
    "/scenes/" + encodeURIComponent(sceneId) + "/ar-track",
    { method: "POST", body: form }
  );
  const data = await safeJson(response, "AR tracking failed");
  applyARGuidance(data);
  trackingIteration += 1;
}

async function continuousTrackingLoop() {
  if (!continuousTracking) return;
  try {
    await trackLockedTargetOnce();
  } catch (error) {
    document.getElementById("cameraStatus").innerHTML =
      "<span class=\"warn\">追跡失敗: " + error.message + "</span>";
  }
  if (continuousTracking) {
    window.setTimeout(continuousTrackingLoop, mobileViewport ? 180 : 120);
  }
}

async function startContinuousTracking() {
  if (!cameraStreamIsLive()) await startCamera();
  if (!lockedTargetId) await locateInCamera();

  continuousTracking = true;
  trackingIteration = 0;
  trackingSessionId = (
    globalThis.crypto && typeof globalThis.crypto.randomUUID === "function"
      ? globalThis.crypto.randomUUID()
      : "track-" + Date.now()
  );
  document.getElementById("cameraTrackBtn").disabled = true;
  document.getElementById("cameraTrackStopBtn").disabled = false;
  continuousTrackingLoop();
}

function stopContinuousTracking(quiet = false) {
  continuousTracking = false;
  trackingSessionId = null;
  const startButton = document.getElementById("cameraTrackBtn");
  const stopButton = document.getElementById("cameraTrackStopBtn");
  if (startButton) startButton.disabled = false;
  if (stopButton) stopButton.disabled = true;
  if (!quiet) {
    const status = document.getElementById("cameraStatus");
    if (status) status.innerHTML +=
      "<br><span class=\"muted\">追跡停止</span>";
  }
}

document.getElementById("cameraTrackBtn").onclick = () =>
  startContinuousTracking().catch(e => {
    document.getElementById("cameraStatus").innerHTML =
      "<span class=\"warn\">" + e.message + "</span>";
    stopContinuousTracking();
  });

document.getElementById("cameraTrackStopBtn").onclick = () =>
  stopContinuousTracking();

document.getElementById("query").addEventListener("input", () => {
  lockedTargetId = null;
  trackingSessionId = null;
});


async function rememberRelation() {
  const parentId = document.getElementById("parentEntity").value;
  const transcript = document.getElementById("relationMemo").value.trim();
  const relation = document.getElementById("relationType").value;
  if (!parentId) throw new Error("親Entityを選択してください");
  if (!transcript) throw new Error("記憶内容を入力してください");

  const response = await fetch(
    "/scenes/" + encodeURIComponent(sceneId) + "/relations/remember",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        parent_entity_id: parentId,
        transcript: transcript,
        relation: relation,
        kind: "object",
      }),
    }
  );
  const data = await safeJson(response, "relation memory failed");

  document.getElementById("relationStatus").innerHTML =
    "<span class=\"ok\">記憶しました: " + data.name + "</span><br>" +
    "parent: " + parentId + " / relation: " + relation;
  await refresh(data.id);
}

document.getElementById("relationMemoryBtn").onclick = () =>
  rememberRelation().catch(e => {
    document.getElementById("relationStatus").innerHTML =
      "<span class=\"warn\">" + e.message + "</span>";
  });

function focusMemoryTarget(target){
  if(geometryVariant==='photo-room' && photoRoom.loaded){
    photoRoom.inside(camera,controls,vec(target.position));
    document.getElementById('photoRoomState').textContent='実写テクスチャ付きモデル · '+target.name+' の場所を表示中';
  }else{controls.target.copy(vec(target.position));controls.update();}
}

function disposeRoute() {
  if (!routeLine) return;
  scene.remove(routeLine);
  routeLine.geometry?.dispose();
  routeLine.material?.dispose();
  routeLine = null;
}

function resourceSnapshot() {
  const positions = pointCloud?.geometry?.getAttribute("position");
  const boundsSize = viewerBounds
    ? viewerBounds.getSize(new THREE.Vector3())
    : null;
  const boundsCenter = viewerBounds
    ? viewerBounds.getCenter(new THREE.Vector3())
    : null;
  let gridRoomUpAlignment = null;
  if (gridHelper && viewerRoomFrame?.axis_y_up) {
    const gridUp = new THREE.Vector3(0, 1, 0)
      .applyQuaternion(gridHelper.quaternion)
      .normalize();
    gridRoomUpAlignment = Math.abs(
      gridUp.dot(vec(viewerRoomFrame.axis_y_up).normalize())
    );
  }
  return {
    mobileViewport,
    photoRoomLoaded:photoRoom.loaded,
    photoRoomSurfaces:photoRoom.group?.children.length||0,
    geometryVariant,
    pointCloudLoaded,
    pointCloudQuality,
    pointCloudMaxPoints: pointCloudBudgets[pointCloudQuality],
    pointCount: positions ? positions.count : 0,
    spatialModelLoaded,
    spatialModelRevision,
    spatialRenderMode,
    roomFrameSnapEnabled: !!viewerRoomFrame?.snap_enabled,
    roomFrameConfidence: viewerRoomFrame?.confidence ?? null,
    roomFrame: viewerRoomFrame ? {
      origin: viewerRoomFrame.origin,
      axisX: viewerRoomFrame.axis_x,
      axisY: viewerRoomFrame.axis_y_up,
      axisZ: viewerRoomFrame.axis_z,
    } : null,
    spatialComponentCount: spatialComponentMeshes.size,
    spatialComponents: [...spatialComponentMeshes.values()].map(mesh => ({
      id: mesh.userData.component?.id || null,
      kind: mesh.userData.component?.kind || null,
      source: mesh.userData.component?.source || null,
      targetAxis: mesh.userData.component?.metadata?.target_axis || null,
      snapApplied: mesh.userData.component?.metadata?.snap_applied ?? null,
      grounded: mesh.userData.component?.metadata?.grounded ?? null,
      center: {
        x: mesh.position.x,
        y: mesh.position.y,
        z: mesh.position.z,
      },
    })),
    spatialShellVertices:
      spatialShell?.geometry?.getAttribute("position")?.count ?? 0,
    spatialShellTriangles:
      spatialShell?.geometry?.index
        ? Math.floor(spatialShell.geometry.index.count / 3)
        : Math.floor((spatialShell?.geometry?.getAttribute("position")?.count ?? 0) / 3),
    viewerBounds: viewerBounds ? {
      min: {
        x: viewerBounds.min.x,
        y: viewerBounds.min.y,
        z: viewerBounds.min.z,
      },
      max: {
        x: viewerBounds.max.x,
        y: viewerBounds.max.y,
        z: viewerBounds.max.z,
      },
      size: {
        x: boundsSize.x,
        y: boundsSize.y,
        z: boundsSize.z,
      },
    } : null,
    cameraDistanceToScene: boundsCenter
      ? camera.position.distanceTo(boundsCenter)
      : null,
    gridSize: gridHelper?.userData?.size ?? null,
    gridDivisions: gridHelper?.userData?.divisions ?? null,
    gridRoomAligned: gridHelper?.userData?.roomAligned ?? false,
    gridRoomUpAlignment,
    shellSupportLayer: !!spatialShell?.userData?.supportLayer,
    shellOpacity: spatialShell?.material?.opacity ?? null,
    geometries: renderer.info.memory.geometries,
    textures: renderer.info.memory.textures,
    drawCalls: renderer.info.render.calls,
    cameraState,
    cameraTracks: cameraStream ? cameraStream.getTracks().filter(track => track.readyState === "live").length : 0,
    audioTracks: mediaStream ? mediaStream.getTracks().filter(track => track.readyState === "live").length : 0,
    jsHeapBytes: performance.memory?.usedJSHeapSize ?? null,
  };
}

window.__spatialMemoryDebug = {
  resourceSnapshot,
  disposePointCloud,
  loadPointCloud,
  disposeSpatialModel,
  loadSpatialModel,
  suspendForBackground,
  projectWorldToClient: position => {
    const projected = new THREE.Vector3(position.x, position.y, position.z).project(camera);
    const rect = renderer.domElement.getBoundingClientRect();
    return {
      x: rect.left + (projected.x + 1) * 0.5 * rect.width,
      y: rect.top + (1 - projected.y) * 0.5 * rect.height,
      visible: projected.z >= -1 && projected.z <= 1,
    };
  },
  selectionSnapshot: () => ({
    mode: select3dMode,
    selected: selected3dPosition ? {
      x: selected3dPosition.x,
      y: selected3dPosition.y,
      z: selected3dPosition.z,
    } : null,
    source: selected3dSource?.type || null,
    componentId: selected3dSource?.component?.id || null,
    componentKind: selected3dSource?.component?.kind || null,
    marker: !!selectionMarker,
  }),
  resumeAfterBackground: () => {
    viewerSuspended = false;
    resumeAnimation();
    if (spatialModelPath || geometryVariant==='photo-room') scheduleSpatialModelLoad();
    if (pointCloudRevision) schedulePointCloudLoad();
  },
};

function suspendForBackground() {
  viewerSuspended = true;
  stopContinuousTracking(true);
  stopCameraStream();
  stopAudioResources();
  disposeSpatialModel();
  disposePointCloud();
  photoRoom.clear();
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
}

function disposeViewer() {
  if (viewerDisposed) return;
  viewerDisposed = true;
  suspendForBackground();
  clearEntities();
  disposeRoute();
  disposeSelectionMarker();
  walker.dispose();
  controls.dispose();
  renderer.dispose();
  if (typeof renderer.forceContextLoss === "function") renderer.forceContextLoss();
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    suspendForBackground();
    return;
  }
  if (!viewerDisposed && (["room", "find"].includes(document.body.dataset.screen) || geometryVariant==='photo-room')) {
    viewerSuspended = false;
    resumeAnimation();
    if (spatialModelPath || geometryVariant==='photo-room') scheduleSpatialModelLoad();
    if (pointCloudRevision) schedulePointCloudLoad();
  }
});

window.addEventListener("pagehide", disposeViewer);
window.addEventListener("pageshow", event => {
  if (event.persisted) window.location.reload();
});

const workspaceScreens = new Set(["room", "find", "memory", "hidden", "debug"]);

function setWorkspaceScreen(name) {
  if (walker.active) setWalking(false);
  if (!workspaceScreens.has(name)) name = "room";
  if (!["room","memory"].includes(name) && select3dMode) set3dSelectMode(false);
  document.body.dataset.screen = name;
  for (const panel of document.querySelectorAll("[data-screen-panel]")) {
    panel.classList.toggle("active", panel.dataset.screenPanel === name);
  }
  for (const button of document.querySelectorAll("[data-screen-target]")) {
    button.classList.toggle("active", button.dataset.screenTarget === name);
    button.setAttribute("aria-pressed", String(button.dataset.screenTarget === name));
  }

  const viewerVisible = name === "room" || name === "find" || (geometryVariant==='photo-room' && ['memory','hidden'].includes(name));
  document.getElementById('registrationTools').hidden=!(name==='room'||(name==='memory'&&geometryVariant==='photo-room'));
  if(name==='memory')requestAnimationFrame(()=>{
    loadAudioMemoryReport().catch(()=>{});
    previewSceneFrame().catch(error=>{document.getElementById('labelStatus').textContent=error.message;});
  });
  if (name !== "find") {
    stopContinuousTracking(true);
    stopCameraStream();
  }
  if (viewerVisible) {
    viewerSuspended = false;
    resumeAnimation();
    if (spatialModelPath || geometryVariant==='photo-room') scheduleSpatialModelLoad();
    if (pointCloudRevision) schedulePointCloudLoad();
    requestAnimationFrame(resize);
  } else {
    disposeSpatialModel();
    disposePointCloud();
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
  }
}

function screenFromHash() {
  const name = location.hash.replace(/^#/, "");
  return workspaceScreens.has(name) ? name : "room";
}

for (const button of document.querySelectorAll("[data-screen-target]")) {
  button.addEventListener("click", () => {
    const target = button.dataset.screenTarget;
    if (location.hash === "#" + target) setWorkspaceScreen(target);
    else location.hash = target;
  });
}

const debugNavButton = document.getElementById("debugNavBtn");
if (debugNavButton) {
  debugNavButton.addEventListener("click", () => {
    location.hash = "debug";
  });
}
window.addEventListener("hashchange", () => setWorkspaceScreen(screenFromHash()));
setWorkspaceScreen(screenFromHash());
