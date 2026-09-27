import * as THREE from 'three';
export function openingPreview(data){
 const root=document.getElementById('viewerLoading'),host=document.getElementById('openingScene');
 let renderer;
 try{renderer=new THREE.WebGLRenderer({alpha:true,antialias:true});}catch{return {photo(){},progress(){},dispose(){}};}
 renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.outputColorSpace=THREE.SRGBColorSpace;host.append(renderer.domElement);
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(38,1,.01,10000);camera.up.set(0,0,1);
 const group=new THREE.Group();scene.add(group);const targets=new Map(),owned=[];
 for(const s of data.surfaces){
  if(s.kind==='ceiling')continue;
  const o=new THREE.Vector3(...s.origin),u=new THREE.Vector3(...s.u),v=new THREE.Vector3(...s.v);
  const normal=new THREE.Vector3().crossVectors(u,v).normalize();
  const front=s.kind==='wall' && normal.dot(new THREE.Vector3(1,-1,0))>0;
  const floor=data.surfaces.find(x=>x.kind==='floor');if(s.kind==='object'&&floor&&Math.abs(normal.z)>.9&&u.length()*v.length()>Math.abs(floor.u[0]*floor.v[1])*.6)continue;
  let positions,uv,indices;
  if(s.polygon){const poly=s.polygon.map(p=>new THREE.Vector2(...p));positions=poly.flatMap(p=>[p.x,p.y,o.z]);uv=poly.flatMap(p=>[(p.x-o.x)/u.x,(p.y-o.y)/v.y]);indices=THREE.ShapeUtils.triangulateShape(poly,[]).flat();}
  else{positions=[o,o.clone().add(u),o.clone().add(u).add(v),o.clone().add(v)].flatMap(p=>p.toArray());uv=[0,0,1,0,1,1,0,1];indices=[0,1,2,0,2,3];}
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));g.setIndex(indices);
  const m=new THREE.MeshBasicMaterial({color:0x163852,transparent:true,opacity:.22,side:THREE.DoubleSide});const mesh=new THREE.Mesh(g,m);if(!front)group.add(mesh);owned.push(g,m);
  const eg=new THREE.EdgesGeometry(g),em=new THREE.LineBasicMaterial({color:0x6dbbff,transparent:true,opacity:.35});group.add(new THREE.LineSegments(eg,em));owned.push(eg,em);
  if(!targets.has(s.texture))targets.set(s.texture,[]);if(!front)targets.get(s.texture).push(m);
 }
 const box=new THREE.Box3().setFromObject(group),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());const radius=Math.max(size.length(),1);group.position.sub(center);
 function resize(){const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;const distance=radius*.85/Math.min(1,camera.aspect);camera.position.set(distance,-distance,distance*.85);camera.lookAt(0,0,0);camera.updateProjectionMatrix();renderer.render(scene,camera);}
 const observer=new ResizeObserver(resize);observer.observe(host);resize();
 document.getElementById('openingStepShape').dataset.done='true';document.getElementById('openingStepPhoto').setAttribute('aria-current','step');
 return {photo(path,texture){for(const m of targets.get(path)||[]){texture.colorSpace=THREE.SRGBColorSpace;m.map=texture;m.color.set(0xffffff);m.opacity=.86;m.needsUpdate=true;}renderer.render(scene,camera);},progress(done,total){const bar=document.getElementById('openingProgress');bar.max=total||1;bar.value=done;document.getElementById('openingCount').textContent=total?`${Math.round(done/total*100)}%`:'準備中';if(done===total){document.getElementById('openingStepPhoto').dataset.done='true';document.getElementById('openingStepPhoto').removeAttribute('aria-current');document.getElementById('openingStepReady').setAttribute('aria-current','step');}},dispose(){observer.disconnect();for(const o of owned)o.dispose();renderer.dispose();renderer.domElement.remove();}};
}
