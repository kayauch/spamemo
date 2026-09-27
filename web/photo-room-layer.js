import * as THREE from 'three';
import {RoomNavigation} from './room-navigation.js';

// The API, observations and camera relocalization all use the original VGGT
// frame. Transform the room once, so hits, entity pins and routes share it.
export class PhotoRoomFrame {
  constructor(data, sceneId) {
    if(data.schema_version !== 2 || data.transform?.scene_id !== sceneId)
      throw Error('この部屋に対応する写真付きモデルではありません');
    const rows=data.transform.generation_world_to_vg_world;
    if(!Array.isArray(rows)||rows.length!==4||rows.some(r=>!Array.isArray(r)||r.length!==4)||!rows.flat().every(Number.isFinite))
      throw Error('部屋の座標変換が不正です');
    this.toWorld=new THREE.Matrix4().set(...rows.flat());
    if(Math.abs(this.toWorld.determinant())<1e-12 || rows[3].some((v,i)=>Math.abs(v-(i===3?1:0))>1e-8))
      throw Error('部屋の座標変換を逆算できません');
    this.toRoom=this.toWorld.clone().invert();
    this.scale=new THREE.Vector3().setFromMatrixColumn(this.toWorld,0).length();
    const basis=[0,1,2].map(i=>new THREE.Vector3().setFromMatrixColumn(this.toWorld,i));
    if(basis.some(v=>Math.abs(v.length()/this.scale-1)>1e-5) ||
       Math.abs(basis[0].dot(basis[1]))>this.scale**2*1e-5 ||
       Math.abs(basis[1].dot(basis[2]))>this.scale**2*1e-5 ||
       Math.abs(basis[0].dot(basis[2]))>this.scale**2*1e-5)
      throw Error('部屋の座標変換の尺度が一致しません');
    this.up=new THREE.Vector3(0,0,1).transformDirection(this.toWorld);
  }
  world(point){return new THREE.Vector3(point.x,point.y,point.z).applyMatrix4(this.toWorld);}
  room(point){return new THREE.Vector3(point.x,point.y,point.z).applyMatrix4(this.toRoom);}
  constrain(navigation,position,delta){
    const start=this.room(position),end=this.room(new THREE.Vector3().copy(position).add(delta));
    return this.world(navigation.move(start,end.sub(start)));
  }
}

export function surfaceGeometry(s) {
  const origin=new THREE.Vector3(...s.origin),u=new THREE.Vector3(...s.u),v=new THREE.Vector3(...s.v);
  let positions,uvs,indices;
  if(s.polygon){
    const polygon=s.polygon.map(p=>new THREE.Vector2(...p));
    positions=polygon.flatMap(p=>[p.x,p.y,s.origin[2]]);
    uvs=polygon.flatMap(p=>[(p.x-origin.x)/u.x,(p.y-origin.y)/v.y]);
    indices=THREE.ShapeUtils.triangulateShape(polygon,[]).flat();
  }else{
    positions=[origin,origin.clone().add(u),origin.clone().add(u).add(v),origin.clone().add(v)].flatMap(p=>p.toArray());
    uvs=[0,0,1,0,1,1,0,1];indices=[0,1,2,0,2,3];
  }
  if(!positions.every(Number.isFinite)||!uvs.every(Number.isFinite))throw Error('写真付きモデルの面が不正です');
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uvs,2));
  geometry.setIndex(indices);geometry.computeVertexNormals();return geometry;
}

function disposeGroup(group){
  group?.traverse(mesh=>{mesh.geometry?.dispose();mesh.material?.map?.dispose();mesh.material?.dispose();});
  group?.removeFromParent();
}

export class PhotoRoomLayer {
  constructor(scene,sceneId,renderer){
    this.scene=scene;this.sceneId=sceneId;this.renderer=renderer;
    this.generation=0;this.group=null;this.pending=null;this.data=null;
  }
  configure(data,base){
    if(!Array.isArray(data.surfaces)||!data.surfaces.length)throw Error('写真付きモデルに面がありません');
    this.frame=new PhotoRoomFrame(data,this.sceneId);
    this.navigation=new RoomNavigation(data.navigation);
    this.data=data;this.base=base;
  }
  get loaded(){return !!this.group;}
  clear(){
    this.generation++;disposeGroup(this.group);this.group=null;this.pending=null;
  }
  load(){
    if(this.loaded)return Promise.resolve(true);
    if(this.pending)return this.pending;
    const generation=this.generation,group=new THREE.Group();
    group.matrixAutoUpdate=false;group.matrix.copy(this.frame.toWorld);
    const task=(async()=>{
      try{
        const loader=new THREE.TextureLoader();
        for(const surface of this.data.surfaces){
          if(generation!==this.generation)return false;
          const texture=await loader.loadAsync(new URL(surface.texture,this.base).href);
          if(generation!==this.generation){texture.dispose();return false;}
          texture.colorSpace=THREE.SRGBColorSpace;
          texture.anisotropy=Math.min(8,this.renderer.capabilities.getMaxAnisotropy());
          let geometry;
          try{geometry=surfaceGeometry(surface);}catch(error){texture.dispose();throw error;}
          const mesh=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide}));
          mesh.userData=surface;group.add(mesh);
        }
        if(generation!==this.generation)return false;
        this.scene.add(group);group.updateMatrixWorld(true);this.group=group;
        this.bounds=new THREE.Box3().setFromObject(group);return true;
      }finally{
        if(this.group!==group)disposeGroup(group);
        if(this.pending===task)this.pending=null;
      }
    })();
    this.pending=task;return task;
  }
  setOverview(enabled){
    for(const mesh of this.group?.children||[])mesh.visible=!(enabled&&['wall','ceiling'].includes(mesh.userData.kind));
  }
  hit(raycaster){
    this.group?.updateMatrixWorld(true);
    return raycaster.intersectObjects((this.group?.children||[]).filter(m=>m.visible),false)[0]||null;
  }
  inside(camera,controls,target=null){
    if(!this.group)return;
    this.setOverview(false);
    const nav=this.navigation,preferred=this.data.navigation.spawn?.slice(0,2);
    const spawn=nav.findSpawn(preferred);if(!spawn)return;
    camera.position.copy(this.frame.world(spawn));
    let focal=target?new THREE.Vector3().copy(target):null;
    if(!focal){
      const objects=new Map();
      for(const mesh of this.group.children){
        if(mesh.userData.object_id==null||['curtain','illumination','chandelier'].includes(mesh.userData.object_label))continue;
        const id=mesh.userData.object_id;
        if(!objects.has(id))objects.set(id,new THREE.Box3());
        objects.get(id).expandByObject(mesh);
      }
      let largest=0;
      for(const box of objects.values()){
        const size=box.getSize(new THREE.Vector3()),volume=size.x*size.y*size.z;
        if(volume>largest){largest=volume;focal=box.getCenter(new THREE.Vector3());}
      }
    }
    if(!focal || focal.distanceTo(camera.position)<.01*this.frame.scale)
      focal=this.frame.world({x:spawn.x,y:spawn.y+1,z:spawn.z-.1});
    controls.target.copy(focal);controls.update();
  }
  overview(camera,controls){
    if(!this.bounds)return;this.setOverview(true);
    const box=new THREE.Box3();
    for(const mesh of this.group.children)if(mesh.userData.kind==='floor')box.expandByObject(mesh);
    const center=box.getCenter(new THREE.Vector3()),distance=box.getSize(new THREE.Vector3()).length();
    const offset=new THREE.Vector3(.55,-.75,.8).transformDirection(this.frame.toWorld).multiplyScalar(distance);
    camera.position.copy(center).add(offset);controls.target.copy(center);controls.update();
  }
}
