import * as THREE from 'three';
import {PLYLoader} from './vendor/PLYLoader.js';
import {OrbitControls} from './vendor/OrbitControls.js';
import {surfaceGeometry} from './photo-room-layer.js';

export class GenerationPreview {
  constructor(root,note){
    this.root=root;this.note=note;this.scene=new THREE.Scene();this.camera=new THREE.PerspectiveCamera(42,1,.01,1000);
    this.renderer=new THREE.WebGLRenderer({alpha:true,antialias:true});this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));root.append(this.renderer.domElement);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableZoom=false;this.controls.enablePan=false;
    this.controls.addEventListener('change',()=>this.draw());this.version=0;this.group=new THREE.Group();this.scene.add(this.group);
    this.resize=new ResizeObserver(()=>{const w=root.clientWidth,h=root.clientHeight;if(!w||!h)return;this.renderer.setSize(w,h);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.draw();});this.resize.observe(root);
    window.addEventListener('pagehide',()=>{this.version++;this.clear();this.resize.disconnect();this.controls.dispose();this.renderer.dispose();},{once:true});
  }
  draw(){this.renderer.render(this.scene,this.camera);}
  clear(){for(const mesh of [...this.group.children]){mesh.geometry?.dispose();mesh.material?.map?.dispose();mesh.material?.dispose();this.group.remove(mesh);}}
  fit(zUp=false){
    const bounds=new THREE.Box3().setFromObject(this.group);if(bounds.isEmpty())return;
    const center=bounds.getCenter(new THREE.Vector3()),distance=Math.max(.1,bounds.getSize(new THREE.Vector3()).length());
    this.camera.up.set(0,zUp?0:1,zUp?1:0);this.camera.position.copy(center).add(new THREE.Vector3(.4,zUp?-.55:.5,zUp?.5:.55).multiplyScalar(distance));
    this.camera.near=distance/1000;this.camera.far=distance*20;this.camera.updateProjectionMatrix();this.controls.target.copy(center);this.controls.update();this.draw();
  }
  async update(job){
    const id=job.scene_id||job.scene;if(!id)return;
    const key=[id,job.review_url||'',job.result_url||''].join('|');if(key===this.key)return;this.key=key;
    const version=++this.version,local=path=>{const u=new URL(path,location.origin);if(u.origin!==location.origin||!u.pathname.startsWith('/static/'))throw Error('Invalid preview URL');return u;};
    try{
      if(job.state==='complete'&&job.result_url){
        const url=local(job.result_url),run=url.searchParams.get('run'),result=url.searchParams.get('result');
        if(![run,result].every(s=>s&&/^[\w-]+$/.test(s)))throw Error('Invalid result');
        const base=`/static/generation-reviews/${run}/${result}/`,response=await fetch(base+'room.json');if(!response.ok)throw Error('Model unavailable');
        const data=await response.json();if(version!==this.version)return;this.clear();
        const loader=new THREE.TextureLoader(),surfaces=data.surfaces.filter(s=>!['ceiling'].includes(s.kind));
        // Show the real model immediately, then add its photos with bounded concurrency.
        const meshes=surfaces.map(s=>{const mesh=new THREE.Mesh(surfaceGeometry(s),new THREE.MeshBasicMaterial({color:s.kind==='wall'?0x92a5b5:0xbecbd5,side:THREE.DoubleSide,transparent:s.kind==='wall',opacity:s.kind==='wall'?.22:1}));this.group.add(mesh);return mesh;});
        this.fit(true);this.note.textContent='作成した部屋のプレビュー · ドラッグで回転';let next=0;
        await Promise.all(Array.from({length:6},async()=>{while(next<surfaces.length&&version===this.version){const i=next++;try{const texture=await loader.loadAsync(base+surfaces[i].texture);if(version!==this.version){texture.dispose();return;}texture.colorSpace=THREE.SRGBColorSpace;meshes[i].material.map=texture;meshes[i].material.color.set(0xffffff);meshes[i].material.needsUpdate=true;this.draw();}catch{/* Retain the visible geometry when an individual photo is unavailable. */}}}));
      }else{
        const geometry=await new PLYLoader().loadAsync(`/scenes/${encodeURIComponent(id)}/point-cloud?max_points=12000`);
        if(version!==this.version){geometry.dispose();return;}this.clear();
        const points=new THREE.Points(geometry,new THREE.PointsMaterial({size:.012,color:0x90cafa,transparent:true,opacity:.8}));this.group.add(points);
        let zUp=false;
        if(job.review_url){
          try{const url=local(job.review_url),run=url.searchParams.get('run'),prediction=url.searchParams.get('prediction');if(![run,prediction].every(s=>s&&/^[\w-]+$/.test(s)))throw Error('Invalid review');
            const response=await fetch(`/static/generation-reviews/${run}/${prediction}/review.json`);if(!response.ok)throw Error('Review unavailable');const report=await response.json();if(version!==this.version)return;
            const rows=report.transform?.vg_world_to_generation_world;if(rows?.flat().length===16){points.applyMatrix4(new THREE.Matrix4().set(...rows.flat()));zUp=true;}
            if(zUp)for(const box of report.prediction?.boxes||[]){if(box.hidden)continue;const geom=new THREE.BoxGeometry(...box.scale.map(v=>Math.max(.02,v))),mesh=new THREE.Mesh(geom,new THREE.MeshBasicMaterial({color:box.class==='wall'?0x84b5d8:0x2699ff,wireframe:true,transparent:true,opacity:.65}));mesh.position.fromArray(box.center);const r=box.rotation;mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(...r[0],0,...r[1],0,...r[2],0,0,0,0,1));this.group.add(mesh);}
          }catch{/* Point cloud remains a truthful intermediate preview. */}
        }
        if(version!==this.version)return;this.fit(zUp);this.note.textContent='撮影から復元した形状 · 推定中のプレビュー';
      }
    }catch{if(version===this.version){this.key=null;this.note.textContent='3Dプレビューを準備しています';}}
  }
}
