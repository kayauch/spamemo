import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import * as GS from './vendor/gaussian-splats-3d.module.js';
import {WalkControls} from './walk-controls.js';

const status = document.querySelector('#status');
const run = new URLSearchParams(location.search).get('run') || 'bedroom-001';
const button = document.querySelector('#walk');
try {
  if (!/^[a-zA-Z0-9_-]+$/.test(run)) throw new Error('無効な結果名です');
  const base = `/static/colmap-reviews/${run}`;
  const response = await fetch(`${base}/review.json`);
  if (!response.ok) throw new Error('この比較結果はまだ準備できていません');
  const review = await response.json();
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, .001, 1000);
  const renderer = new THREE.WebGLRenderer({antialias:false});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.setSize(innerWidth, innerHeight); document.body.append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  const walker = new WalkControls(camera, renderer.domElement, () => controls, () => stop());
  const up = document.querySelector('#up');
  const requestedUp = new URLSearchParams(location.search).get('up');
  if (['image-top','image-left','image-right','image-bottom'].includes(requestedUp)) up.value=requestedUp;
  function stop() { walker.stop(); button.textContent = '中を歩く'; }
  function selectView(index) {
    stop();
    const view = review.cameras[index];
    const r = view.rotation;
    camera.position.fromArray(view.position);
    const sideways = ['image-left','image-right'].includes(up.value);
    const axis = sideways ? 0 : 1;
    const sign = ['image-top','image-left'].includes(up.value) ? -1 : 1;
    camera.up.set(sign*r[0][axis], sign*r[1][axis], sign*r[2][axis]);
    controls.target.copy(camera.position).add(new THREE.Vector3(r[0][2], r[1][2], r[2][2]));
    camera.fov = THREE.MathUtils.radToDeg(2*Math.atan(sideways ? view.width/(2*view.fx) : view.height/(2*view.fy)));
    camera.updateProjectionMatrix(); camera.lookAt(controls.target); controls.update();
  }
  const views = document.querySelector('#views');
  review.cameras.forEach((v, i) => views.add(new Option(`撮影位置 ${i+1}`, i)));
  views.onchange = () => selectView(Number(views.value));
  up.onchange = () => selectView(Number(views.value));
  views.value = String(Math.floor(review.cameras.length/2));
  selectView(Number(views.value));
  const splats = new GS.DropInViewer({splatRenderMode:GS.SplatRenderMode.ThreeD,
    sharedMemoryForWorkers:false,gpuAcceleratedSort:false,integerBasedSort:false,
    sphericalHarmonicsDegree:2,sceneRevealMode:GS.SceneRevealMode.Instant});
  await splats.addSplatScene(`${base}/model.ply`, {format:GS.SceneFormat.Ply,showLoadingUI:false,splatAlphaRemovalThreshold:1});
  scene.add(splats);
  const bounds = new THREE.Box3().setFromPoints(review.cameras.map(v => new THREE.Vector3().fromArray(v.position)));
  button.disabled = false;
  button.onclick = () => { if(walker.active) stop(); else {walker.start(Math.max(bounds.getSize(new THREE.Vector3()).length(),1));button.textContent='歩行を終了';} };
  status.textContent = `VGGT不使用・登録画像 ${review.registered_images}/${review.total_images}枚`;
  document.querySelector('#comparison').href = `${base}/comparison.jpg`;
  renderer.setAnimationLoop(now => {walker.update(now);if(!walker.active) controls.update();renderer.render(scene,camera);});
  addEventListener('resize', () => {camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
} catch(error) {status.textContent=error.message;}
