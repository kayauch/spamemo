import * as THREE from './vendor/three.module.js';

// Separate overlays leave shared furniture textures and opacity untouched.
export function createStorageHighlight(scene){
 const group=new THREE.Group();group.name='storage-selection';scene.add(group);
 function clear(){for(const child of [...group.children]){child.geometry.dispose();child.material.dispose();group.remove(child);}}
 function show(meshes=[],fallback=null){
  clear();const bounds=new THREE.Box3(),seen=new Set();
  for(const root of meshes){root.updateWorldMatrix(true,true);root.traverse(mesh=>{
   if(!mesh.isMesh||seen.has(mesh)||!mesh.geometry)return;seen.add(mesh);
   const geometry=mesh.geometry.clone();
   const overlay=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:0x35b8ff,transparent:true,opacity:.22,depthTest:false,depthWrite:false,side:THREE.DoubleSide}));
   overlay.matrixAutoUpdate=false;overlay.matrix.copy(mesh.matrixWorld);overlay.renderOrder=30;group.add(overlay);
   bounds.expandByObject(mesh);
  });}
  if(bounds.isEmpty()&&fallback){
   const {room_position,width,depth,height,angle=0}=fallback;
   if(Array.isArray(room_position)&&room_position.every(Number.isFinite)&&[width,depth,height].every(v=>Number.isFinite(v)&&v>0)){
    const box=new THREE.Mesh(new THREE.BoxGeometry(width/100,depth/100,height/100),new THREE.MeshBasicMaterial({color:0x35b8ff,transparent:true,opacity:.2,depthTest:false,depthWrite:false}));
    box.position.fromArray(room_position);box.position.z+=height/200;box.rotation.z=angle*Math.PI/180;box.renderOrder=30;group.add(box);box.updateWorldMatrix(true,false);bounds.expandByObject(box);
   }
  }
  if(!bounds.isEmpty()){
   const frame=new THREE.Box3Helper(bounds.clone().expandByScalar(.018),0x7bdbff);frame.material.depthTest=false;frame.material.depthWrite=false;frame.material.transparent=true;frame.material.opacity=.95;frame.renderOrder=31;group.add(frame);
  }
  return bounds;
 }
 function dispose(){clear();group.removeFromParent();}
 return {show,clear,dispose};
}
